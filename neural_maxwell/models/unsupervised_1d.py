import numpy as np
import torch
import torch.nn as nn

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Simulation1D
from neural_maxwell.utils import tensor_diff


class MaxwellDense(nn.Module):

    def __init__(self, size = 64, src_x = 32, supervised = False, drop_p = 0.1, regularize_A_phi = True):
        super().__init__()

        self.size = size
        self.src_x = src_x
        self.supervised = supervised
        self.regularize_A_phi = regularize_A_phi
        self.drop_p = drop_p

        self.layer_dims = [self.size, 128, 256, 256, 256, 128, self.size]

        layers_amp = []
        layers_phi = []
        for i in range(len(self.layer_dims) - 1):
            layers_amp.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            layers_phi.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))

        self.layers_amp = nn.ModuleList(layers_amp)
        self.layers_phi = nn.ModuleList(layers_phi)

    def forward_amplitude_phase(self, x):
        A = x
        imax = len(self.layers_amp) - 1
        for i, layer in enumerate(self.layers_amp):
            A = layer(A)
            if i < imax:
                A = nn.ReLU()(A)
                A = nn.Dropout(p = self.drop_p)(A)
            else:
                A = nn.ELU()(A) + 1 + 0.1

        phi = x
        imax = len(self.layers_phi) - 1
        for i, layer in enumerate(self.layers_phi):
            phi = layer(phi)
            if i < imax:
                phi = nn.ReLU()(phi)
                phi = nn.Dropout(p = 0.05)(phi)
            else:
                phi = 2 * np.pi * nn.Tanh()(phi)

        return A, phi

    def get_fields(self, epsilons, add_zero_bc = False, A_phi = None):
        # Get amplitude and phase vectors
        if A_phi is None:
            A, phi = self.forward_amplitude_phase(epsilons)
        else:
            A, phi = A_phi

        # Combine to form waveform
        x = (PIXEL_SIZE * (torch.arange(self.size, dtype = torch.float, device = device) - self.src_x))
        fields = A * torch.cos(OMEGA_1550 / C * torch.sqrt(epsilons) * x + phi)

        if add_zero_bc:
            batch_size, _ = epsilons.shape
            zero = torch.zeros((batch_size, 1), device = device)
            fields = torch.cat([zero, fields, zero], dim = -1)

        return fields

    def forward(self, epsilons):
        # Compute Ez fields
        A, phi = self.forward_amplitude_phase(epsilons)
        fields = self.get_fields(epsilons, A_phi = (A, phi))

        if self.supervised:
            labels = torch.empty_like(fields)
            for i, perm in enumerate(epsilons.detach().numpy()):
                _, _, _, _, Ez = Simulation1D(buffer_length = 16).solve(perm, omega = OMEGA_1550)
                labels[i, :] = torch.tensor(np.imag(Ez[16:-16])).float()
            return fields - labels

        else:
            batch_size, _ = epsilons.shape

            # Add zero field amplitudes at edge points for resonator BC's
            zero = torch.zeros((batch_size, 1), device = device)
            E = torch.cat([zero, fields, zero], dim = -1)

            # Add first layer of cavity BC's
            barrier = torch.full((batch_size, 1), -1e10, device = device)
            eps = torch.cat([barrier, epsilons, barrier], dim = -1)

            # Compute Maxwell operator on fields
            diffs = tensor_diff(E, n = 2, padding = None)
            curl_curl_E = (SCALE / PIXEL_SIZE ** 2) * torch.cat([zero, diffs, zero], dim = -1)
            epsilon_E = (SCALE * -OMEGA_1550 ** 2 * MU0 * EPSILON0) * eps * E

            # Compute free-current vector
            J = torch.zeros_like(E)
            J[:, self.src_x + 1] = 1.526814027933079

            out = curl_curl_E - epsilon_E - J

            # Penalize excessive variation in A/phi
            if self.regularize_A_phi:
                A_variation = torch.sum(torch.abs(tensor_diff(A)), -1, keepdim = True)
                phi_variation = torch.sum(torch.abs(tensor_diff(phi)), -1, keepdim = True)
                eps_variation = torch.sum(torch.abs(tensor_diff(epsilons)), -1, keepdim = True)
                factor = 1e-1 / (1 + eps_variation)
                out = torch.cat([out, factor * A_variation, factor * phi_variation], dim = -1)

            REMOVE_ENDS = True
            if REMOVE_ENDS:
                out = out[:, 1:-1]

            return out
