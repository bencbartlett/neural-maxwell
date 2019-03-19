import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Cavity1D
from neural_maxwell.utils import tensor_diff


class MaxwellConvolutionalMiniNet(nn.Module):

    def __init__(self, size = 64, src_x = 32, supervised = False):
        super().__init__()

        self.size = size
        self.src_x = src_x
        self.supervised = supervised

        self.convnet0 = self.make_convolutional_mininet(1, [128, 256, 128, 1], [3, 3, 3, 3])  # 16 in, 8 out
        self.convnet1 = self.make_convolutional_mininet(2, [128, 256, 128, 1], [5, 5, 5, 5])  # 32 in, 16 out
        self.convnet2 = self.make_convolutional_mininet(2, [256, 512, 256, 1], [9, 9, 9, 9])  # 64 in, 32 out

        self.layer_dims = [64 - 2 - 4 - 6, self.size]

        layers_amp = []
        layers_phi = []
        for i in range(len(self.layer_dims) - 1):
            layers_amp.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            layers_phi.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1], bias = False))
        self.layers_amp = nn.ModuleList(layers_amp)
        self.layers_phi = nn.ModuleList(layers_phi)

    @staticmethod
    def make_convolutional_mininet(num_input_channels, channels, kernel_sizes):
        layers = []
        in_channels = num_input_channels
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 0))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward_convolutional(self, eps):

        in0 = F.interpolate(eps, scale_factor = 1 / 4)
        in1 = F.interpolate(eps, scale_factor = 1 / 2)
        in2 = eps

        out0 = self.convnet0(in0)

        in1 = torch.cat((in1, F.interpolate(out0, scale_factor = 4)), 1)
        out1 = self.convnet1(in1)

        in2 = torch.cat((in2, F.interpolate(out1, scale_factor = 4)), 1)
        out2 = self.convnet2(in2)

        out = out2.view(batch_size, 32)

        return out

    def forward_amplitude_phase(self, epsilons):
        # Get convolutional output
        x = self.forward_convolutional(epsilons)

        A = x
        for i, layer in enumerate(self.layers_amp):
            A = layer(A)
            if i < len(self.layers_amp) - 1:
                A = nn.ReLU()(A)

        phi = x
        for i, layer in enumerate(self.layers_phi):
            phi = layer(phi)
            if i < len(self.layers_amp) - 1:
                phi = nn.ReLU()(phi)

        return A, phi

    def get_fields(self, epsilons):
        # Get amplitude and phase vectors
        A, phi = self.forward_amplitude_phase(epsilons)

        # Combine to form waveform
        x = (PIXEL_SIZE * (torch.arange(self.size, dtype = torch.float, device = device) - self.src_x))
        fields = A * torch.cos(OMEGA / C * torch.sqrt(epsilons) * x + phi)

        return fields

    def forward(self, epsilons):
        # Compute Ez fields
        fields = self.get_fields(epsilons)

        if self.supervised:
            labels = torch.empty_like(fields)
            for i, perm in enumerate(epsilons.detach().numpy()):
                _, _, _, _, Ez = Cavity1D(buffer_length = 16).solve(perm, omega = OMEGA_1550)
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
            epsilon_E = (SCALE * -OMEGA ** 2 * MU0 * EPSILON0) * eps * E

            # Compute free-current vector
            J = torch.zeros_like(E)
            J[:, self.src_x + 1] = 1.526814027933079

            out = curl_curl_E - epsilon_E - J

            return out
