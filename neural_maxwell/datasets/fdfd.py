import numpy as np
import scipy.sparse as sp
from angler import Simulation
from angler.derivatives import unpack_derivs

from neural_maxwell.constants import GRID_SIZE, OMEGA_1550, EPSILON0, MU0

DEVICE_LENGTH = 64
NPML = 16
NPML_BUFFER = 32
TOTAL_LENGTH = DEVICE_LENGTH + 2 * NPML_BUFFER + 2 * NPML
CLIPPED_LENGTH = TOTAL_LENGTH - 2 * NPML


class Simulation1D:
    '''FDFD simulation of a 1-dimensional system'''

    def __init__(self, mode = "Ez", device_length = DEVICE_LENGTH, npml = NPML, npml_buffer = NPML_BUFFER, dl = 0.05,
                 L0 = 1e-6):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.npml_buffer = npml_buffer
        self.dl = dl
        self.L0 = L0

    def solve(self, epsilons: np.array, omega = OMEGA_1550):
        total_length = self.device_length + 2 * self.npml_buffer + 2 * self.npml
        start = self.npml + self.npml_buffer
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        vac_perm = np.ones((2, total_length), dtype = np.float64)

        perms = np.copy(vac_perm)
        perms[:, start:end] = epsilons

        # vac_sim = Simulation(self.omega, vac_perm, self.dl, [self.npml, 0], self.mode, L0=self.L0)
        # vac_sim.src[pos_x, pos_y] = 1
        # Hx_vac, Hy_vac, Ez_vac = vac_sim.solve_fields()

        # src_x = int(self.npml + self.npml_buffer // 2)
        # src_x = np.random.randint(self.npml+1, self.npml + self.npml_buffer)
        src_x = self.npml + 16

        sim = Simulation(omega, perms, self.dl, [0, self.npml], self.mode, L0 = self.L0)
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
    '''FDFD simulation of a 1-dimensional cavity system'''

    def __init__(self, mode = "Ez", device_length = 65, npml = 0, cavity_buffer = 16, buffer_permittivity = -1e20,
                 dl = 0.05, L0 = 1e-6):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.cavity_buffer = cavity_buffer
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = L0

    def solve(self, epsilons: np.array, omega = OMEGA_1550, src_x = None):

        total_length = self.device_length + 2 * self.cavity_buffer + 2 * self.npml
        start = self.npml + self.cavity_buffer
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        perms = np.ones((2, total_length), dtype = np.float64)

        # set permittivity and reflection zone
        perms[:, :start] = self.buffer_permittivity
        perms[:, start:end] = epsilons
        perms[:, end:] = self.buffer_permittivity

        if src_x is None:
            src_x = int(self.device_length / 2)

        sim = Simulation(omega, perms, self.dl, [0, self.npml], self.mode, L0 = self.L0)
        sim.src[:, src_x + self.npml + self.cavity_buffer] = 1j

        clip0 = None  # self.npml + self.cavity_buffer
        clip1 = None  # -(self.npml + self.cavity_buffer)

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

    def get_operators(self, omega = OMEGA_1550):

        total_length = self.device_length + 2 * self.cavity_buffer + 2 * self.npml

        perms = np.ones(total_length, dtype = np.float64)

        start = self.npml + self.cavity_buffer
        end = start + self.device_length

        perms[:start] = self.buffer_permittivity
        perms[end:] = self.buffer_permittivity

        sim = Simulation(omega, perms, self.dl, [0, self.npml], self.mode, L0 = self.L0)

        Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

        N = np.asarray(perms.shape)
        M = np.prod(N)

        vector_eps_z = EPSILON0 * self.L0 * perms.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format = 'csr')

        curl_curl = (Dxf @ Dxb + Dyf @ Dyb)

        other = omega ** 2 * MU0 * self.L0 * T_eps_z

        return curl_curl.todense(), other.todense()


class Simulation2D:
    '''FDFD simulation of a 2-dimensional system'''

    def __init__(self, epsilons: np.ndarray, omega = OMEGA_1550 * 2, source_pos = None, mode = "Ez",
                 grid_size = GRID_SIZE, npml = 16, npml_buffer = 16, dl = 0.05, L0 = 1e-6):
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

        vac_perm = np.ones((self.total_grid_size, self.total_grid_size), dtype = np.float64)

        perms = np.copy(vac_perm)
        perms[start:end, start:end] = self.epsilons

        pos_x, pos_y = self.source_pos + start

        vac_sim = Simulation(self.omega, vac_perm, self.dl, [self.npml, self.npml], self.mode, L0 = self.L0)
        vac_sim.src[pos_x, pos_y] = 1
        Hx_vac, Hy_vac, Ez_vac = vac_sim.solve_fields()

        sim = Simulation(self.omega, perms, self.dl, [self.npml, self.npml], self.mode, L0 = self.L0)
        sim.src[pos_x, pos_y] = 1
        Hx, Hy, Ez = sim.solve_fields()

        self.fields = {"Hx": Hx[start:end, start:end],
                       "Hy": Hy[start:end, start:end],
                       "Ez": Ez[start:end, start:end]}
        self.fields_vac = {"Hx": Hx_vac[start:end, start:end],
                           "Hy": Hy_vac[start:end, start:end],
                           "Ez": Ez_vac[start:end, start:end]}

    def get_proximity_matrix(self, mode = "inv_squared"):
        start = self.npml + self.npml_buffer
        end = start + self.grid_size
        x0, y0 = self.source_pos + start
        if mode == "linear":
            return np.sqrt([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                            for y in range(self.total_grid_size)], dtype = np.float64)[start:end, start:end]
        if mode == "squared":
            return np.array([[(x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                             for y in range(self.total_grid_size)], dtype = np.float64)[start:end, start:end]
        if mode == "inv_linear":
            return 1 / np.sqrt([[1 + (x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                                for y in range(self.total_grid_size)], dtype = np.float64)[start:end, start:end]
        if mode == "inv_squared":
            return 1 / np.array([[1 + (x - x0) ** 2 + (y - y0) ** 2 for x in range(self.total_grid_size)]
                                 for y in range(self.total_grid_size)], dtype = np.float64)[start:end, start:end]


class Cavity2D:
    '''FDFD simulation of a 2-dimensional cavity system'''

    def __init__(self, mode = "Ez", device_length = 32, npml = 0, cavity_buffer = 4, buffer_permittivity = -1e20,
                 dl = 0.05, L0 = 1e-6):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.cavity_buffer = cavity_buffer
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = L0

    def solve(self, epsilons: np.array, omega = OMEGA_1550, src_x = None, src_y = None):

        total_length = self.device_length + 2 * self.cavity_buffer + 2 * self.npml
        start = self.npml + self.cavity_buffer
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

        clip0 = None  # self.npml + self.cavity_buffer
        clip1 = None  # -(self.npml + self.cavity_buffer)

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

        total_length = self.device_length + 2 * self.cavity_buffer + 2 * self.npml

        perms = np.ones((total_length, total_length), dtype = np.float64)

        start = self.npml + self.cavity_buffer
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
