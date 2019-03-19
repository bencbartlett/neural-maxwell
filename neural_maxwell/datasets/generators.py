import h5py
import numpy as np

from neural_maxwell.constants import GRID_SIZE, OMEGA_1550, eps_si
from neural_maxwell.datasets.fdfd import Simulation1D
from neural_maxwell.utils import pbar

DEVICE_LENGTH = 64
NPML = 16
NPML_BUFFER = 32
TOTAL_LENGTH = DEVICE_LENGTH + 2 * NPML_BUFFER + 2 * NPML
CLIPPED_LENGTH = TOTAL_LENGTH - 2 * NPML


def create_dataset(f, N, name, s = CLIPPED_LENGTH):
    grp = f.require_group(name)
    epsilons = grp.require_dataset("epsilons", (N, s), dtype = np.float64)
    src = grp.require_dataset("src", (N,), dtype = np.int16)
    Hx = grp.require_dataset("Hx", (N, s), dtype = np.complex128)
    Hy = grp.require_dataset("Hy", (N, s), dtype = np.complex128)
    Ez = grp.require_dataset("Ez", (N, s), dtype = np.complex128)

    dataset = {
        "epsilons": epsilons,
        "src"     : src,
        "Hx"      : Hx,
        "Hy"      : Hy,
        "Ez"      : Ez,
    }

    return dataset


def make_batch(permmitivity_generator, filename, dsname, N = 1000, omega = OMEGA_1550, npml = NPML):
    f = h5py.File(filename, "a")
    ds = create_dataset(f, N, dsname)

    for i in pbar(range(N)):
        perms = permmitivity_generator()
        sim = Simulation1D(npml = npml)
        epsilons, src, Hx, Hy, Ez = sim.solve(perms, omega = omega)

        ds["epsilons"][i] = epsilons
        ds["src"][i] = src
        ds["Hx"][i] = Hx
        ds["Hy"][i] = Hy
        ds["Ez"][i] = Ez


def load_batch(filename, batchname, load_src = True):
    f = h5py.File(filename)
    ds = f[batchname]
    dataset = {
        "epsilons": ds["epsilons"],
        "Hx"      : ds["Hx"],
        "Hy"      : ds["Hy"],
        "Ez"      : ds["Ez"],
    }
    if load_src:
        dataset["src"] = ds["src"]
    return dataset


class PermittivityGenerators1D:

    @staticmethod
    def random(s = DEVICE_LENGTH, eps_max = eps_si):
        return (eps_max - 1) * np.random.rand(s) + 1

    @staticmethod
    def binary(s = DEVICE_LENGTH, eps = eps_si):
        epsilons = np.ones(s)
        epsilons += (eps - 1) * np.round(np.random.rand(s))
        return epsilons

    @staticmethod
    def alternating_layers(s = DEVICE_LENGTH, num_layers = 5, eps1 = eps_si, eps2 = 1.0):
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

    @staticmethod
    def random_number_alternating_layers(s = DEVICE_LENGTH, num_layers_range = (2, 10), eps1 = eps_si, eps2 = 1.0):
        num_layers = np.random.randint(num_layers_range[0], num_layers_range[1] + 1)
        return PermittivityGenerators1D.alternating_layers(s, num_layers, eps1 = eps1, eps2 = eps2)

    @staticmethod
    def random_layers(s = DEVICE_LENGTH, num_layers = 5, eps_range = (1, 4.5 ** 2)):
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

    @staticmethod
    def random_number_random_layers(s = DEVICE_LENGTH, num_layers_range = (2, 20), eps_range = (1, 4.5 ** 2)):
        num_layers = np.random.randint(num_layers_range[0], num_layers_range[1] + 1)
        return PermittivityGenerators1D.random_layers(s, num_layers, eps_range = eps_range)


class PermittivityGenerators2D:

    @staticmethod
    def perm_random(s = GRID_SIZE):
        return eps_si * np.random.rand(s, s)

    @staticmethod
    def perm_rectangle(s = GRID_SIZE):
        p_matrix = np.ones((s, s))
        x0, y0 = np.random.randint(16, s - 16, 2)
        dx, dy = np.random.randint(5, 16, 2)
        p_matrix[x0:x0 + dx, y0:y0 + dy] = eps_si
        return p_matrix

    @staticmethod
    def perm_ellipse(s = GRID_SIZE):
        p_matrix = np.ones((s, s))
        x0, y0 = np.random.randint(16, s - 16, 2)
        rx, ry = np.random.randint(5, 16, 2)

        x, y = np.meshgrid(np.arange(s), np.arange(s))
        ellipse = ((x - x0) / rx) ** 2 + ((y - y0) / ry) ** 2 <= 1
        p_matrix[ellipse < 1.0] = eps_si
        return p_matrix

# def get_A_ops_1d(epsilons, npml, omega = OMEGA_1550, dl = 0.05, L0 = 1e-6):
#     sim = Simulation(omega, epsilons, dl, [npml, 0], "Ez", L0 = L0)
#
#     Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)
#
#     N = np.asarray(epsilons.shape)
#     M = np.prod(N)
#
#     vector_eps_z = EPSILON0 * L0 * epsilons.reshape((-1,))
#     T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format = 'csr')
#
#     curl_curl = (Dxf @ Dxb + Dyf @ Dyb)
#
#     other = omega ** 2 * MU0 * L0 * T_eps_z
#
#     return curl_curl.todense(), other.todense()
#
#
# def get_A_ops_2d(epsilons, npml, omega = OMEGA_1550, dl = 0.05, L0 = 1e-6):
#     sim = Simulation(omega, epsilons, dl, [npml, npml], "Ez", L0 = L0)
#
#     Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)
#
#     N = np.asarray(epsilons.shape)
#     M = np.prod(N)
#
#     vector_eps_z = EPSILON0 * L0 * epsilons.reshape((-1,))
#     T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format = 'csr')
#
#     curl_curl = (Dxf @ Dxb + Dyf @ Dyb)
#
#     other = omega ** 2 * MU0 * L0 * T_eps_z
#
#     return curl_curl.todense(), other.todense()

# def create_dataset(f, N, name, s = GRID_SIZE):
#     grp = f.require_group(name)
#     epsilons = grp.require_dataset("epsilons", (N, s, s), dtype = np.float64)
#     proximities = grp.require_dataset("proximities", (N, s, s), dtype = np.float64)
#
#     Hx = grp.require_dataset("Hx", (N, s, s), dtype = np.complex128)
#     Hy = grp.require_dataset("Hy", (N, s, s), dtype = np.complex128)
#     Ez = grp.require_dataset("Ez", (N, s, s), dtype = np.complex128)
#
#     Hx_vac = grp.require_dataset("Hx_vac", (N, s, s), dtype = np.complex128)
#     Hy_vac = grp.require_dataset("Hy_vac", (N, s, s), dtype = np.complex128)
#     Ez_vac = grp.require_dataset("Ez_vac", (N, s, s), dtype = np.complex128)
#
#     dataset = {
#         "epsilons"   : epsilons,
#         "proximities": proximities,
#         "Hx"         : Hx,
#         "Hy"         : Hy,
#         "Ez"         : Ez,
#         "Hx_vac"     : Hx_vac,
#         "Hy_vac"     : Hy_vac,
#         "Ez_vac"     : Ez_vac
#     }
#
#     return dataset


# def make_batch(permmitivity_generator, name, N = 1000, omega = OMEGA_1550 * 2):
#     f = h5py.File("datasets/test.hdf5", "a")
#     ds = create_dataset(f, N, name)
#
#     for i in pbar(range(N)):
#         epsilons = permmitivity_generator()
#         sim = Simulation1D(epsilons, omega = omega)
#         sim.solve()
#
#         ds["epsilons"][i] = sim.epsilons
#         ds["proximities"][i] = sim.get_proximity_matrix("inv_squared")
#         ds["Hx"][i] = sim.fields["Hx"]
#         ds["Hy"][i] = sim.fields["Hy"]
#         ds["Ez"][i] = sim.fields["Ez"]
#         ds["Hx_vac"][i] = sim.fields_vac["Hx"]
#         ds["Hy_vac"][i] = sim.fields_vac["Hy"]
#         ds["Ez_vac"][i] = sim.fields_vac["Ez"]

# def load_batch(filename, batchname):
#     f = h5py.File(filename)
#     ds = f[batchname]
#     dataset = {
#         "epsilons"   : ds["epsilons"],
#         "proximities": ds["proximities"],
#         "Hx"         : ds["Hx"],
#         "Hy"         : ds["Hy"],
#         "Ez"         : ds["Ez"],
#         "Hx_vac"     : ds["Hx_vac"],
#         "Hy_vac"     : ds["Hy_vac"],
#         "Ez_vac"     : ds["Ez_vac"]
#     }
#     return dataset
