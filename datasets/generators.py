import h5py
import numpy as np
from angler import Simulation

from datasets.settings import eps_si, pbar


def make_simulation(permittivities: np.ndarray):
    '''
    Create a simulation for an embedded 64x64 device permittivity matrix inside a 128x128 vacuum matrix
    :param permittivities: 64x64 matrix of permittivity values
    :return: Hx, Hy, Ez
    '''

    omega = 1.215e15  # 1550nm frequency
    dl = 0.02  # grid size (units of L0, which defaults to 1e-6)
    NPML = [15, 15]  # number of pml grid points on x and y borders
    simulation = Simulation(omega, permittivities, dl, NPML, 'Ez')

    # Add a source
    simulation.src[16, 16] = 1

    return simulation.solve_fields()


def get_proximity_matrix():
    '''Returns squared distance values from source'''
    x0, y0 = 16, 16
    return np.array([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(64)] for y in range(64)], dtype = np.float64)


def create_dataset(f, N, name):
    grp = f.require_group(name)
    permittivities = grp.require_dataset("permittivities", (N, 64, 64), dtype = np.float64)
    proximities = grp.require_dataset("proximities", (N, 64, 64), dtype = np.float64)
    Hx_ds = grp.require_dataset("Hx", (N, 64, 64), dtype = np.complex128)
    Hy_ds = grp.require_dataset("Hy", (N, 64, 64), dtype = np.complex128)
    Ez_ds = grp.require_dataset("Ez", (N, 64, 64), dtype = np.complex128)
    return permittivities, proximities, Hx_ds, Hy_ds, Ez_ds


def make_random_permittivity_batch(N = 1000):
    '''
    Creates simulations of continuous-random permittivity values ranging from vacuum to silicon
    :param N:
    :return:
    '''

    f = h5py.File("datasets/test.hdf5", "a")
    permittivities, proximities, Hx_ds, Hy_ds, Ez_ds = create_dataset(f, N, "random")
    prox = get_proximity_matrix()


    for i in pbar(range(N)):
        p_matrix = eps_si * np.random.rand(64, 64)
        Hx, Hy, Ez = make_simulation(p_matrix)
        permittivities[i] = p_matrix
        proximities[i] = prox
        Hx_ds[i] = Hx
        Hy_ds[i] = Hy
        Ez_ds[i] = Ez


def make_random_rectangle_batch(N = 1000):

    f = h5py.File("datasets/test.hdf5", "a")
    permittivities, proximities, Hx_ds, Hy_ds, Ez_ds = create_dataset(f, N, "rectangles")
    prox = get_proximity_matrix()

    for i in pbar(range(N)):
        p_matrix = np.ones((64, 64))
        x1, x2 = sorted(16 + np.random.randint(0, 32, 2))
        y1, y2 = sorted(16 + np.random.randint(0, 32, 2))
        p_matrix[x1:x2, y1:y2] = eps_si

        Hx, Hy, Ez = make_simulation(p_matrix)
        permittivities[i] = p_matrix
        proximities[i] = prox
        Hx_ds[i] = Hx
        Hy_ds[i] = Hy
        Ez_ds[i] = Ez

def make_random_ellipse_batch(N = 1000):

    f = h5py.File("datasets/test.hdf5", "a")
    permittivities, proximities, Hx_ds, Hy_ds, Ez_ds = create_dataset(f, N, "ellipses")
    prox = get_proximity_matrix()

    for i in pbar(range(N)):
        p_matrix = np.ones((64, 64))
        x0, y0 = np.random.randint(16,48,2)
        rx, ry = np.random.randint(5,32,2)

        # Setup arrays which just list the x and y coordinates
        x, y = np.meshgrid(np.arange(64), np.arange(64))

        # Calculate the ellipse values all at once
        ellipse = ((x - x0) / rx) ** 2 + ((y - y0) / ry) ** 2 <= 1

        p_matrix[ellipse < 1.0] = eps_si

        Hx, Hy, Ez = make_simulation(p_matrix)
        permittivities[i] = p_matrix
        proximities[i] = prox
        Hx_ds[i] = Hx
        Hy_ds[i] = Hy
        Ez_ds[i] = Ez


def load_batch(filename, batchname):
    f = h5py.File(filename)
    grp = f[batchname]
    return grp["permittivities"], grp["proximities"], grp["Hx"], grp["Hy"], grp["Ez"]
