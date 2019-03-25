import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import torch
import torch.nn.functional as F
from angler import Simulation
from angler.derivatives import unpack_derivs

from neural_maxwell.constants import *


def maxwell_residual_1d(fields, epsilons, curl_curl_op,
                        buffer_length = BUFFER_LENGTH, buffer_permittivity = BUFFER_PERMITTIVITY,
                        add_buffer = True, trim_buffer = True):
    '''Compute ∇×∇×E - omega^2 mu0 epsilon E'''

    batch_size, _ = epsilons.shape

    # Add zero field amplitudes at edge points for resonator BC's
    if add_buffer:
        fields = F.pad(fields, [buffer_length] * 2)
    fields = fields.view(batch_size, -1, 1)

    # Add first layer of cavity BC's
    if add_buffer:
        epsilons = F.pad(epsilons, [buffer_length] * 2, "constant", buffer_permittivity)
    epsilons = epsilons.view(batch_size, -1, 1)

    # Compute Maxwell operator on fields
    curl_curl_E = (SCALE / L0 ** 2) * torch.matmul(curl_curl_op, fields).view(batch_size, -1, 1)
    epsilon_E = (SCALE * -OMEGA_1550 ** 2 * MU0 * EPSILON0) * epsilons * fields

    out = curl_curl_E - epsilon_E

    if trim_buffer and buffer_length > 0:
        return out[:, buffer_length:-buffer_length]
    else:
        return out


def maxwell_residual_2d(fields, epsilons, curl_curl_op,
                        buffer_length = BUFFER_LENGTH, buffer_permittivity = BUFFER_PERMITTIVITY,
                        add_buffer = True, trim_buffer = True):
    '''Compute ∇×∇×E - omega^2 mu0 epsilon E'''

    batch_size, W, H = epsilons.shape

    # Add zero field amplitudes at edge points for resonator BC's
    if add_buffer:
        fields = F.pad(fields, [buffer_length] * 4)
        W += 2 * buffer_length
        H += 2 * buffer_length
    fields = fields.view(batch_size, -1, 1)

    # Add first layer of cavity BC's
    if add_buffer:
        epsilons = F.pad(epsilons, [buffer_length] * 4, "constant", buffer_permittivity)
    epsilons = epsilons.view(batch_size, -1, 1)

    # Compute Maxwell operator on fields
    curl_curl_E = (SCALE / L0 ** 2) * torch.matmul(curl_curl_op, fields).view(batch_size, -1, 1)
    epsilon_E = (SCALE * -OMEGA_1550 ** 2 * MU0 * EPSILON0) * epsilons * fields

    out = curl_curl_E - epsilon_E
    out = out.view(batch_size, W, H)

    if trim_buffer and buffer_length > 0:
        return out[:, buffer_length:-buffer_length, buffer_length:-buffer_length]
    else:
        return out


def maxwell_residual_2d_tf(fields, epsilons, curl_curl_op,
                           buffer_length = BUFFER_LENGTH, buffer_permittivity = BUFFER_PERMITTIVITY,
                           add_buffer = True, trim_buffer = True):
    '''Compute ∇×∇×E - omega^2 mu0 epsilon E'''

    _, W, H = epsilons.shape.as_list()
    dim = tf.reduce_prod(tf.shape(epsilons)[1:])

    # Add zero field amplitudes at edge points for resonator BC's
    if add_buffer:
        fields = tf.pad(fields, [[0,0]] + [[buffer_length, buffer_length]] * 2, mode = "CONSTANT")
        W += 2 * buffer_length
        H += 2 * buffer_length
    fields = tf.reshape(fields, (-1, dim))

    # Add first layer of cavity BC's
    if add_buffer:
        epsilons = tf.pad(epsilons, [[0,0]] + [[buffer_length, buffer_length]] * 2, mode = "CONSTANT",
                          constant_values = buffer_permittivity)
    epsilons = tf.reshape(epsilons, (-1, dim))

    # Compute Maxwell operator on fields
    curl_curl_E = (SCALE / L0 ** 2) * tf.matmul(curl_curl_op, fields)
    curl_curl_E = tf.reshape(curl_curl_E, (-1, dim))
    epsilon_E = (SCALE * -OMEGA_1550 ** 2 * MU0 * EPSILON0) * epsilons * fields

    out = curl_curl_E - epsilon_E
    out = tf.reshape(out, (-1, W, H))

    if trim_buffer and buffer_length > 0:
        return out[:, buffer_length:-buffer_length, buffer_length:-buffer_length]
    else:
        return out


class Simulation1D:
    '''FDFD simulation of a 1-dimensional system'''

    def __init__(self, mode = "Ez", device_length = DEVICE_LENGTH, npml = 0, buffer_length = BUFFER_LENGTH,
                 buffer_permittivity = BUFFER_PERMITTIVITY, dl = dL, l0 = L0, use_dirichlet_bcs = False):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = l0
        self.use_dirichlet_bcs = use_dirichlet_bcs

    def solve(self, epsilons: np.array, omega = OMEGA_1550, src_x = None, clip_buffers = False):

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

        sim = Simulation(omega, permittivities, self.dl, [self.npml, 0], self.mode, L0 = self.L0,
                         use_dirichlet_bcs = self.use_dirichlet_bcs)
        sim.src[src_x + self.npml + self.buffer_length] = 1j

        if clip_buffers:
            clip0 = self.npml + self.buffer_length
            clip1 = -(self.npml + self.buffer_length)
        else:
            clip0 = None
            clip1 = None

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

        sim = Simulation(omega, perms, self.dl, [self.npml, 0], self.mode, L0 = self.L0,
                         use_dirichlet_bcs = self.use_dirichlet_bcs)

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

    def __init__(self, mode = "Ez", device_length = DEVICE_LENGTH_2D, npml = 0, buffer_length = BUFFER_LENGTH,
                 buffer_permittivity = BUFFER_PERMITTIVITY, dl = dL, l0 = L0, use_dirichlet_bcs = False):
        self.mode = mode
        self.device_length = device_length
        self.npml = npml
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.dl = dl
        self.L0 = l0
        self.use_dirichlet_bcs = use_dirichlet_bcs

    def solve(self, epsilons: np.array, omega = OMEGA_1550, src_x = None, src_y = None, clip_buffers = False):

        total_length = self.device_length + 2 * self.buffer_length + 2 * self.npml
        start = self.npml + self.buffer_length
        end = start + self.device_length

        # need to use two rows to avoid issues with fd-derivative operators
        permittivities = np.ones((total_length, total_length), dtype = np.float64)

        # set permittivity and reflection zone
        permittivities[:, :start] = self.buffer_permittivity
        permittivities[:start, :] = self.buffer_permittivity

        permittivities[start:end, start:end] = epsilons

        permittivities[:, end:] = self.buffer_permittivity
        permittivities[end:, :] = self.buffer_permittivity

        if src_x is None:
            src_x = self.device_length // 2
        if src_y is None:
            src_y = self.device_length // 2

        sim = Simulation(omega, permittivities, self.dl, [self.npml, self.npml], self.mode, L0 = self.L0,
                         use_dirichlet_bcs = self.use_dirichlet_bcs)
        sim.src[src_y + self.npml + self.buffer_length, src_x + self.npml + self.buffer_length] = 1j

        if clip_buffers:
            clip0 = self.npml + self.buffer_length
            clip1 = -(self.npml + self.buffer_length)
        else:
            clip0 = None
            clip1 = None

        if self.mode == "Ez":
            Hx, Hy, Ez = sim.solve_fields()
            permittivities = permittivities[clip0:clip1, clip0:clip1]
            Hx = Hx[clip0:clip1, clip0:clip1]
            Hy = Hy[clip0:clip1, clip0:clip1]
            Ez = Ez[clip0:clip1, clip0:clip1]
            return permittivities, src_x, src_y, Hx, Hy, Ez

        elif self.mode == "Hz":
            Ex, Ey, Hz = sim.solve_fields()
            permittivities = permittivities[clip0:clip1, clip0:clip1]
            Ex = Ex[clip0:clip1, clip0:clip1]
            Ey = Ey[clip0:clip1, clip0:clip1]
            Hz = Hz[clip0:clip1, clip0:clip1]
            return permittivities, src_x, src_y, Ex, Ey, Hz

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

        sim = Simulation(omega, perms, self.dl, [self.npml, self.npml], self.mode, L0 = self.L0,
                         use_dirichlet_bcs = self.use_dirichlet_bcs)

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
