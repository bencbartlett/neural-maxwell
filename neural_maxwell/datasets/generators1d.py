import h5py
import numpy as np
import scipy.sparse as sp
from angler import Simulation
from angler.derivatives import unpack_derivs

from neural_maxwell.constants import OMEGA_1550, eps_si, EPSILON0, MU0
from neural_maxwell.utils import pbar

DEVICE_LENGTH = 64
NPML = 16
NPML_BUFFER = 32
TOTAL_LENGTH = DEVICE_LENGTH + 2 * NPML_BUFFER + 2 * NPML
CLIPPED_LENGTH = TOTAL_LENGTH - 2 * NPML

class Simulation1D:

    def __init__(self, mode="Ez", device_length=DEVICE_LENGTH, npml=NPML, npml_buffer=NPML_BUFFER, dl=0.05, L0=1e-6):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.npml_buffer = npml_buffer
        self.dl = dl
        self.L0 = L0

    def solve(self, epsilons: np.array, omega=OMEGA_1550):
        total_length = self.device_length + 2 * self.npml_buffer + 2 * self.npml
        start = self.npml + self.npml_buffer
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        vac_perm = np.ones((2, total_length), dtype=np.float64)

        perms = np.copy(vac_perm)
        perms[:, start:end] = epsilons

        # vac_sim = Simulation(self.omega, vac_perm, self.dl, [self.npml, 0], self.mode, L0=self.L0)
        # vac_sim.src[pos_x, pos_y] = 1
        # Hx_vac, Hy_vac, Ez_vac = vac_sim.solve_fields()

        # src_x = int(self.npml + self.npml_buffer // 2)
        # src_x = np.random.randint(self.npml+1, self.npml + self.npml_buffer)
        src_x = self.npml + 16

        sim = Simulation(omega, perms, self.dl, [0, self.npml], self.mode, L0=self.L0)
        sim.src[:, src_x] = 1

        clip0 = self.npml  # + self.npml_buffer
        clip1 = -self.npml

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            if self.npml > 0:
                src_x = src_x - self.npml
                perms = perms[:, clip0:clip1]
                Hx = Hx[:, clip0:clip1]
                Hy = Hy[:, clip0:clip1]
                Ez = Ez[:, clip0:clip1]
            return perms[0], src_x, Hx[0], Hy[0], Ez[0]

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            if self.npml > 0:
                src_x = src_x - self.npml
                perms = perms[:, clip0:clip1]
                Ex = Ex[:, clip0:clip1]
                Ey = Ey[:, clip0:clip1]
                Hz = Hz[:, clip0:clip1]
            return perms[0], src_x, Ex[0], Ey[0], Hz[0]

        else:
            raise ValueError("Polarization must be Ez or Hz!")


class Cavity1D:

    def __init__(self, mode="Ez", device_length=65, npml=0, cavity_buffer=16, buffer_permittivity=-1e20, dl=0.05, L0=1e-6):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.cavity_buffer = cavity_buffer
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = L0

    def solve(self, epsilons: np.array, omega=OMEGA_1550, src_x=None):

        total_length = self.device_length + 2 * self.cavity_buffer + 2 * self.npml
        start = self.npml + self.cavity_buffer
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        perms = np.ones((2, total_length), dtype=np.float64)

        # set permittivity and reflection zone
        perms[:, :start] = self.buffer_permittivity
        perms[:, start:end] = epsilons
        perms[:, end:] = self.buffer_permittivity

        if src_x is None:
            src_x = int(self.device_length / 2)

        sim = Simulation(omega, perms, self.dl, [0, self.npml], self.mode, L0=self.L0)
        sim.src[:, src_x + self.npml + self.cavity_buffer] = 1j

        clip0 = None# self.npml + self.cavity_buffer
        clip1 = None#-(self.npml + self.cavity_buffer)

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            perms = perms[:, clip0:clip1]
            Hx = Hx[:, clip0:clip1]
            Hy = Hy[:, clip0:clip1]
            Ez = Ez[:, clip0:clip1]
            return perms[0], src_x, Hx[0], Hy[0], Ez[0]

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            perms = perms[:, clip0:clip1]
            Ex = Ex[:, clip0:clip1]
            Ey = Ey[:, clip0:clip1]
            Hz = Hz[:, clip0:clip1]
            return perms[0], src_x, Ex[0], Ey[0], Hz[0]

        else:
            raise ValueError("Polarization must be Ez or Hz!")
            
    def get_operators(self, omega=OMEGA_1550):

        total_length = self.device_length + 2 * self.cavity_buffer + 2 * self.npml

        perms = np.ones(total_length, dtype=np.float64)

        start = self.npml + self.cavity_buffer
        end = start + self.device_length

        perms[:start] = self.buffer_permittivity
        perms[end:] = self.buffer_permittivity

        sim = Simulation(omega, perms, self.dl, [0, self.npml], self.mode, L0=self.L0)

        Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

        N = np.asarray(perms.shape) 
        M = np.prod(N) 

        vector_eps_z = EPSILON0 * self.L0 * perms.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format='csr')

        curl_curl = (Dxf@Dxb + Dyf@Dyb)

        other = omega**2 * MU0 * self.L0 * T_eps_z

        return curl_curl.todense(), other.todense()
    
def get_A_ops_1d(epsilons, npml, omega=OMEGA_1550, dl=0.05, L0=1e-6):

    sim = Simulation(omega, epsilons, dl, [npml, 0], "Ez", L0=L0)

    Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

    N = np.asarray(epsilons.shape) 
    M = np.prod(N) 

    vector_eps_z = EPSILON0 * L0 * epsilons.reshape((-1,))
    T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format='csr')

    curl_curl = (Dxf@Dxb + Dyf@Dyb)

    other = omega**2 * MU0 * L0 * T_eps_z

    return curl_curl.todense(), other.todense()


def create_dataset(f, N, name, s=CLIPPED_LENGTH):
    grp = f.require_group(name)
    epsilons = grp.require_dataset("epsilons", (N, s), dtype=np.float64)
    src = grp.require_dataset("src", (N,), dtype=np.int16)
    Hx = grp.require_dataset("Hx", (N, s), dtype=np.complex128)
    Hy = grp.require_dataset("Hy", (N, s), dtype=np.complex128)
    Ez = grp.require_dataset("Ez", (N, s), dtype=np.complex128)

    dataset = {
        "epsilons": epsilons,
        "src": src,
        "Hx": Hx,
        "Hy": Hy,
        "Ez": Ez,
    }

    return dataset


def make_batch(permmitivity_generator, filename, dsname, N=1000, omega=OMEGA_1550, npml=NPML):
    f = h5py.File(filename, "a")
    ds = create_dataset(f, N, dsname)

    for i in pbar(range(N)):
        perms = permmitivity_generator()
        sim = Simulation1D(npml=npml)
        epsilons, src, Hx, Hy, Ez = sim.solve(perms, omega=omega)

        ds["epsilons"][i] = epsilons
        ds["src"][i] = src
        ds["Hx"][i] = Hx
        ds["Hy"][i] = Hy
        ds["Ez"][i] = Ez


def load_batch(filename, batchname, load_src=True):
    f = h5py.File(filename)
    ds = f[batchname]
    dataset = {
        "epsilons": ds["epsilons"],
        "Hx": ds["Hx"],
        "Hy": ds["Hy"],
        "Ez": ds["Ez"],
    }
    if load_src:
        dataset["src"] = ds["src"]
    return dataset


def perm_random(s=DEVICE_LENGTH, eps_max=eps_si):
    return (eps_max - 1) * np.random.rand(s) + 1


def perm_binary(s=DEVICE_LENGTH, eps=eps_si):
    epsilons = np.ones(s)
    epsilons += (eps - 1) * np.round(np.random.rand(s))
    return epsilons


def perm_alternating_layers(s=DEVICE_LENGTH, num_layers=5, eps1=eps_si, eps2=1.0):
    layer_indices = np.random.rand(num_layers)
    layer_indices = np.floor(layer_indices * s / np.sum(layer_indices))
    epsilons = np.ones(s)
    layer_start = 0
    for i, layer_size in enumerate(layer_indices):
        eps = eps1 if i % 2 == 0 else eps2
        layer_end = int(layer_start + layer_size)
        epsilons[layer_start:layer_end] = eps
        layer_start = layer_end
    return epsilons


def perm_random_number_alternating_layers(s=DEVICE_LENGTH, num_layers_range=(2, 10), eps1=eps_si, eps2=1.0):
    num_layers = np.random.randint(num_layers_range[0], num_layers_range[1] + 1)
    return perm_alternating_layers(s, num_layers, eps1=eps1, eps2=eps2)


def perm_random_layers(s=DEVICE_LENGTH, num_layers=5, eps_range=(1, 4.5 ** 2)):
    layer_indices = np.random.rand(num_layers)
    layer_indices = np.floor(layer_indices * s / np.sum(layer_indices))
    epsilons = np.ones(s)
    layer_start = 0
    for i, layer_size in enumerate(layer_indices):
        eps = np.random.uniform(eps_range[0], eps_range[1])
        layer_end = int(layer_start + layer_size)
        epsilons[layer_start:layer_end] = eps
        layer_start = layer_end
    return epsilons


def perm_random_number_random_layers(s=DEVICE_LENGTH, num_layers_range=(2, 20), eps_range=(1, 4.5 ** 2)):
    num_layers = np.random.randint(num_layers_range[0], num_layers_range[1] + 1)
    return perm_random_layers(s, num_layers, eps_range=eps_range)
