import h5py
import numpy as np
from angler import Simulation

from datasets.settings import GRID_SIZE, OMEGA_1550, eps_si, pbar


class SimulationData:

    def __init__(self, epsilons: np.ndarray, omega=OMEGA_1550 * 2, source_pos=None, mode="Ez",
                 grid_size=GRID_SIZE, npml=16, npml_buffer=16, dl=0.05, L0=1e-6):
        self.epsilons = epsilons
        self.omega = omega
        self.mode = mode
        self.grid_size = grid_size
        self.npml = npml
        self.npml_buffer = npml_buffer
        self.total_grid_size = self.grid_size + 2 * self.npml + 2 * self.npml_buffer
        self.dl = dl
        self.L0 = L0
        if source_pos is None:
            pos_x = np.random.randint(0, self.npml_buffer - 1)
            pos_y = np.random.randint(0, self.grid_size)
            self.source_pos = np.array([pos_x, pos_y])
        else:
            self.source_pos = source_pos
        self.fields = None
        self.fields_vac = None
        self.proximities = None

    def solve(self):
        start = self.npml + self.npml_buffer
        end = start + self.grid_size

        vac_perm = np.ones((self.total_grid_size, self.total_grid_size), dtype=np.float64)

        perms = np.copy(vac_perm)
        perms[start:end, start:end] = self.epsilons

        pos_x, pos_y = self.source_pos + start

        vac_sim = Simulation(self.omega, vac_perm, self.dl, [self.npml, self.npml], self.mode, L0=self.L0)
        vac_sim.src[pos_x, pos_y] = 1
        Hx_vac, Hy_vac, Ez_vac = vac_sim.solve_fields()

        sim = Simulation(self.omega, perms, self.dl, [self.npml, self.npml], self.mode, L0=self.L0)
        sim.src[pos_x, pos_y] = 1
        Hx, Hy, Ez = sim.solve_fields()

        self.fields = {"Hx": Hx[start:end, start:end],
                       "Hy": Hy[start:end, start:end],
                       "Ez": Ez[start:end, start:end]}
        self.fields_vac = {"Hx": Hx_vac[start:end, start:end],
                           "Hy": Hy_vac[start:end, start:end],
                           "Ez": Ez_vac[start:end, start:end]}

    def get_proximity_matrix(self, mode="inv_squared"):
        start = self.npml + self.npml_buffer
        end = start + self.grid_size
        x0, y0 = self.source_pos + start
        if mode == "linear":
            return np.sqrt([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                            for y in range(self.total_grid_size)], dtype=np.float64)[start:end, start:end]
        if mode == "squared":
            return np.array([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                             for y in range(self.total_grid_size)], dtype=np.float64)[start:end, start:end]
        if mode == "inv_linear":
            return 1 / np.sqrt([[1 + (x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                                for y in range(self.total_grid_size)], dtype=np.float64)[start:end, start:end]
        if mode == "inv_squared":
            return 1 / np.array([[1 + (x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                                 for y in range(self.total_grid_size)], dtype=np.float64)[start:end, start:end]


# def make_simulation(permittivities: np.ndarray):
#     '''
#     Create a simulation for an embedded 64x64 device permittivity matrix inside a 128x128 vacuum matrix
#     :param permittivities: 64x64 matrix of permittivity values
#     :return: Hx, Hy, Ez
#     '''
#
#     omega = 1.215e15  # 1550nm frequency
#     dl = 0.02  # grid size (units of L0, which defaults to 1e-6)
#     NPML = [15, 15]  # number of pml grid points on x and y borders
#
#     simulation = Simulation(omega, permittivities, dl, NPML, 'Ez')
#
#     # Add a source
#     simulation.src[16, 16] = 1
#
#     return simulation.solve_fields()
#
#
# def get_proximity_matrix():
#     '''Returns squared distance values from source'''
#     x0, y0 = 16, 16
#     return np.array([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(64)] for y in range(64)], dtype = np.float64)


def create_dataset(f, N, name, s=GRID_SIZE):
    grp = f.require_group(name)
    epsilons = grp.require_dataset("epsilons", (N, s, s), dtype=np.float64)
    proximities = grp.require_dataset("proximities", (N, s, s), dtype=np.float64)

    Hx = grp.require_dataset("Hx", (N, s, s), dtype=np.complex128)
    Hy = grp.require_dataset("Hy", (N, s, s), dtype=np.complex128)
    Ez = grp.require_dataset("Ez", (N, s, s), dtype=np.complex128)

    Hx_vac = grp.require_dataset("Hx_vac", (N, s, s), dtype=np.complex128)
    Hy_vac = grp.require_dataset("Hy_vac", (N, s, s), dtype=np.complex128)
    Ez_vac = grp.require_dataset("Ez_vac", (N, s, s), dtype=np.complex128)

    dataset = {
        "epsilons": epsilons,
        "proximities": proximities,
        "Hx": Hx,
        "Hy": Hy,
        "Ez": Ez,
        "Hx_vac": Hx_vac,
        "Hy_vac": Hy_vac,
        "Ez_vac": Ez_vac
    }

    return dataset


def make_batch(permmitivity_generator, name, N=1000):
    f = h5py.File("datasets/test.hdf5", "a")
    ds = create_dataset(f, N, name)

    for i in pbar(range(N)):
        epsilons = permmitivity_generator()
        sim = SimulationData(epsilons)
        sim.solve()

        ds["epsilons"][i] = sim.epsilons
        ds["proximities"][i] = sim.get_proximity_matrix("inv_squared")
        ds["Hx"][i] = sim.fields["Hx"]
        ds["Hy"][i] = sim.fields["Hy"]
        ds["Ez"][i] = sim.fields["Ez"]
        ds["Hx_vac"][i] = sim.fields_vac["Hx"]
        ds["Hy_vac"][i] = sim.fields_vac["Hy"]
        ds["Ez_vac"][i] = sim.fields_vac["Ez"]


def perm_random(s=GRID_SIZE):
    return eps_si * np.random.rand(s, s)


def perm_rectangle(s=GRID_SIZE):
    p_matrix = np.ones((s, s))
    x0, y0 = np.random.randint(16, s - 16, 2)
    dx, dy = np.random.randint(5, 16, 2)
    p_matrix[x0:x0 + dx, y0:y0 + dy] = eps_si
    return p_matrix


def perm_ellipse(s=GRID_SIZE):
    p_matrix = np.ones((s, s))
    x0, y0 = np.random.randint(16, s - 16, 2)
    rx, ry = np.random.randint(5, 16, 2)

    x, y = np.meshgrid(np.arange(s), np.arange(s))
    ellipse = ((x - x0) / rx) ** 2 + ((y - y0) / ry) ** 2 <= 1
    p_matrix[ellipse < 1.0] = eps_si
    return p_matrix


def load_batch(filename, batchname):
    f = h5py.File(filename)
    ds = f[batchname]
    dataset = {
        "epsilons": ds["epsilons"],
        "proximities": ds["proximities"],
        "Hx": ds["Hx"],
        "Hy": ds["Hy"],
        "Ez": ds["Ez"],
        "Hx_vac": ds["Hx_vac"],
        "Hy_vac": ds["Hy_vac"],
        "Ez_vac": ds["Ez_vac"]
    }
    return dataset
