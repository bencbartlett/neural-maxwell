import h5py
import numpy as np
from angler import Simulation

from datasets.settings import OMEGA_1550, eps_si, eps_sio2, pbar

DEVICE_LENGTH = 64
NPML = 16
NPML_BUFFER = 16
TOTAL_LENGTH = DEVICE_LENGTH + 3 * NPML_BUFFER + 2 * NPML
CLIPPED_LENGTH = TOTAL_LENGTH - 2 * NPML - NPML_BUFFER


class Simulation1D:

    def __init__(self, mode="Ez", device_length=DEVICE_LENGTH, npml=NPML, npml_buffer=NPML_BUFFER, dl=0.05, L0=1e-6):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.npml_buffer = npml_buffer
        self.dl = dl
        self.L0 = L0

    def solve(self, epsilons: np.array, omega=OMEGA_1550):
        total_length = self.device_length + 3 * self.npml_buffer + 2 * self.npml
        start = self.npml + 2 * self.npml_buffer
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        vac_perm = np.ones((2, total_length), dtype=np.float64)

        perms = np.copy(vac_perm)
        perms[:, start:end] = epsilons

        # vac_sim = Simulation(self.omega, vac_perm, self.dl, [self.npml, 0], self.mode, L0=self.L0)
        # vac_sim.src[pos_x, pos_y] = 1
        # Hx_vac, Hy_vac, Ez_vac = vac_sim.solve_fields()

        src_x = int(self.npml + self.npml_buffer // 2)

        sim = Simulation(omega, perms, self.dl, [0, self.npml], self.mode, L0=self.L0)
        sim.src[:, src_x] = 1
        Hx, Hy, Ez = sim.solve_fields()

        clip0 = self.npml + self.npml_buffer
        clip1 = -self.npml
        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            return perms[0][clip0:clip1], Hx[0][clip0:clip1], Hy[0][clip0:clip1], Ez[0][clip0:clip1]

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            return perms[0][clip0:clip1], Ex[0][clip0:clip1], Ey[0][clip0:clip1], Ez[0][clip0:clip1]

        else:
            raise ValueError("Polarization must be Ez or Hz!")


def create_dataset(f, N, name, s=CLIPPED_LENGTH):
    grp = f.require_group(name)
    epsilons = grp.require_dataset("epsilons", (N, s), dtype=np.float64)
    Hx = grp.require_dataset("Hx", (N, s), dtype=np.complex128)
    Hy = grp.require_dataset("Hy", (N, s), dtype=np.complex128)
    Ez = grp.require_dataset("Ez", (N, s), dtype=np.complex128)

    dataset = {
        "epsilons": epsilons,
        "Hx": Hx,
        "Hy": Hy,
        "Ez": Ez,
    }

    return dataset


def make_batch(permmitivity_generator, name, N=1000, omega=OMEGA_1550):
    f = h5py.File("datasets/test_1d.hdf5", "a")
    ds = create_dataset(f, N, name)

    for i in pbar(range(N)):
        perms = permmitivity_generator()
        sim = Simulation1D()
        epsilons, Hx, Hy, Ez = sim.solve(perms, omega=omega)

        ds["epsilons"][i] = epsilons
        ds["Hx"][i] = Hx
        ds["Hy"][i] = Hy
        ds["Ez"][i] = Ez

def load_batch(filename, batchname):
    f = h5py.File(filename)
    ds = f[batchname]
    dataset = {
        "epsilons": ds["epsilons"],
        "Hx": ds["Hx"],
        "Hy": ds["Hy"],
        "Ez": ds["Ez"],
    }
    return dataset

def perm_random(s=DEVICE_LENGTH, eps_max=eps_si):
    return (eps_max - 1) * np.random.rand(s) + 1


def perm_binary(s=DEVICE_LENGTH, eps=eps_si):
    epsilons = np.ones(s)
    epsilons += (eps - 1) * np.round(np.random.rand(s))
    return epsilons


def perm_alternating_layers(s=DEVICE_LENGTH, num_layers=5, eps1=eps_si, eps2=eps_sio2):
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


def perm_random_number_alternating_layers(s=DEVICE_LENGTH, num_layers_range=(2, 20), eps1=eps_si, eps2=eps_sio2):
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


def perm_random_number_arandom_layers(s=DEVICE_LENGTH, num_layers_range=(2, 20), eps_range=(1, 4.5 ** 2)):
    num_layers = np.random.randint(num_layers_range[0], num_layers_range[1] + 1)
    return perm_random_layers(s, num_layers, eps_range=eps_range)
