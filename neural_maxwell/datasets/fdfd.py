import numpy as np
import scipy.sparse as sp
from angler import Simulation
from angler.derivatives import unpack_derivs

from neural_maxwell.constants import DEVICE_LENGTH, OMEGA_1550, EPSILON0, MU0, dL, L0, \
    BUFFER_PERMITTIVITY


class Simulation1D:
    '''FDFD simulation of a 1-dimensional system'''

    def __init__(self, mode = "Ez", device_length = DEVICE_LENGTH, npml = 0, buffer_length = 16,
                 buffer_permittivity = BUFFER_PERMITTIVITY, dl = dL, l0 = L0):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = l0

    def solve(self, epsilons: np.array, omega = OMEGA_1550, src_x = None):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml
        start = self.npml + self.buffer_length
        end = start + self.device_length

        permittivities = np.ones(total_length, dtype = np.float64)

        # set permittivity and reflection zone
        permittivities[:start] = self.buffer_permittivity
        permittivities[start:end] = epsilons
        permittivities[end:] = self.buffer_permittivity

        if src_x is None:
            src_x = int(self.device_length / 2)

        sim = Simulation(omega, permittivities, self.dl, [self.npml, 0], self.mode, L0 = self.L0)
        sim.src[src_x + self.npml + self.buffer_length] = 1j

        clip0 = None  # self.npml + self.buffer_length
        clip1 = None  # -(self.npml + self.buffer_length)

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            permittivities = permittivities[clip0:clip1]
            Hx = Hx[clip0:clip1]
            Hy = Hy[clip0:clip1]
            Ez = Ez[clip0:clip1]
            return permittivities, src_x, Hx, Hy, Ez

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            permittivities = permittivities[clip0:clip1]
            Ex = Ex[clip0:clip1]
            Ey = Ey[clip0:clip1]
            Hz = Hz[clip0:clip1]
            return permittivities, src_x, Ex, Ey, Hz

        else:
            raise ValueError("Polarization must be Ez or Hz!")

    def get_operators(self, omega = OMEGA_1550):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml

        perms = np.ones(total_length, dtype = np.float64)

        start = self.npml + self.buffer_length
        end = start + self.device_length

        perms[:start] = self.buffer_permittivity
        perms[end:] = self.buffer_permittivity

        sim = Simulation(omega, perms, self.dl, [self.npml, 0], self.mode, L0 = self.L0)

        Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

        N = np.asarray(perms.shape)
        M = np.prod(N)

        vector_eps_z = EPSILON0 * self.L0 * perms.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format = 'csr')

        curl_curl = (Dxf @ Dxb + Dyf @ Dyb)

        other = omega ** 2 * MU0 * self.L0 * T_eps_z

        return curl_curl.todense(), other.todense()


class Simulation2D:
    '''FDFD simulation of a 2-dimensional  system'''

    def __init__(self, mode = "Ez", device_length = 32, npml = 0, buffer_length = 4,
                 buffer_permittivity = BUFFER_PERMITTIVITY, dl = dL, L0 = L0):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = L0

    def solve(self, epsilons: np.array, omega = OMEGA_1550, src_x = None, src_y = None):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml
        start = self.npml + self.buffer_length
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        perms = np.ones((total_length, total_length), dtype = np.float64)

        # set permittivity and reflection zone
        perms[:, :start] = self.buffer_permittivity
        perms[:start, :] = self.buffer_permittivity

        perms[start:end, start:end] = epsilons

        perms[:, end:] = self.buffer_permittivity
        perms[end:, :] = self.buffer_permittivity

        if src_x is None:
            src_x = total_length // 2
        if src_y is None:
            src_y = total_length // 2

        sim = Simulation(omega, perms, self.dl, [self.npml, self.npml], self.mode, L0 = self.L0)
        sim.src[src_y, src_x] = 1j

        clip0 = None  # self.npml + self.buffer_length
        clip1 = None  # -(self.npml + self.buffer_length)

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            perms = perms[clip0:clip1, clip0:clip1]
            Hx = Hx[clip0:clip1, clip0:clip1]
            Hy = Hy[clip0:clip1, clip0:clip1]
            Ez = Ez[clip0:clip1, clip0:clip1]
            return perms, src_x, src_y, Hx, Hy, Ez

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            perms = perms[clip0:clip1, clip0:clip1]
            Ex = Ex[clip0:clip1, clip0:clip1]
            Ey = Ey[clip0:clip1, clip0:clip1]
            Hz = Hz[clip0:clip1, clip0:clip1]
            return perms, src_x, src_y, Ex, Ey, Hz

        else:
            raise ValueError("Polarization must be Ez or Hz!")

    def get_operators(self, omega = OMEGA_1550):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml

        perms = np.ones((total_length, total_length), dtype = np.float64)

        start = self.npml + self.buffer_length
        end = start + self.device_length

        # set permittivity and reflection zone
        perms[:, :start] = self.buffer_permittivity
        perms[:start, :] = self.buffer_permittivity
        perms[:, end:] = self.buffer_permittivity
        perms[end:, :] = self.buffer_permittivity

        sim = Simulation(omega, perms, self.dl, [self.npml, self.npml], self.mode, L0 = self.L0)

        Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

        N = np.asarray(perms.shape)
        M = np.prod(N)

        vector_eps_z = EPSILON0 * self.L0 * perms.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format = 'csr')

        curl_curl = (Dxf @ Dxb + Dyf @ Dyb)

        other = omega ** 2 * MU0 * self.L0 * T_eps_z

        return curl_curl.todense(), other.todense()

    # def get_proximity_matrix(self, mode = "inv_squared"):
    #     total_grid_size = self.device_length + 2 * self.npml + 2 * self.buffer_length
    #     start = self.npml + self.buffer_length
    #     end = start + self.device_length
    #     x0, y0 = self.source_pos + start
    #     if mode == "linear":
    #         return np.sqrt([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(total_grid_size)]
    #                         for y in range(total_grid_size)], dtype = np.float64)[start:end, start:end]
    #     if mode == "squared":
    #         return np.array([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(total_grid_size)]
    #                          for y in range(total_grid_size)], dtype = np.float64)[start:end, start:end]
    #     if mode == "inv_linear":
    #         return 1 / np.sqrt([[1 + (x - x0) ** 2 + (y - y0) ** 2 for x in range(total_grid_size)]
    #                             for y in range(total_grid_size)], dtype = np.float64)[start:end, start:end]
    #     if mode == "inv_squared":
    #         return 1 / np.array([[1 + (x - x0) ** 2 + (y - y0) ** 2 for x in range(total_grid_size)]
    #                              for y in range(total_grid_size)], dtype = np.float64)[start:end, start:end]
