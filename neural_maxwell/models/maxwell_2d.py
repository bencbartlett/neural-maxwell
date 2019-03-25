import numpy as np
import torch
import torch.nn as nn

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Simulation2D, maxwell_residual_2d
from neural_maxwell.utils import conv_output_size


class MaxwellSolver2D(nn.Module):

    def __init__(self, size = 32, buffer_length=4, buffer_permittivity = BUFFER_PERMITTIVITY, npml = 0, src_x = 16, src_y = 16, channels = None, kernels = None, drop_p = 0.1):
        super().__init__()

        self.size = size
        self.src_x = src_x
        self.src_y = src_y
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.drop_p = drop_p

        self.sim = Simulation2D(device_length = self.size, buffer_length = self.buffer_length, buffer_permittivity=self.buffer_permittivity)
        curl_op, eps_op = self.sim.get_operators()
#         self.curl_curl_op = torch.tensor(np.asarray(np.real(curl_op)), device = device).float()
        self.curl_curl_re = torch.tensor(np.asarray(np.real(curl_op)), device = device).float()
        self.curl_curl_im = torch.tensor(np.asarray(np.imag(curl_op)), device = device).float()

        if channels is None or kernels is None:
            channels = [64] * 7
            kernels = [5] * 7

        layers = []
        in_channels = 1
        out_size = self.size
        for out_channels, kernel_size in zip(channels, kernels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 0))
            layers.append(nn.ReLU())
            if self.drop_p > 0:
                layers.append(nn.Dropout(p = self.drop_p))
            in_channels = out_channels
            out_size = conv_output_size(out_size, kernel_size)

        self.convnet = nn.Sequential(*layers)

        self.densenet = nn.Sequential(
                nn.Linear(out_size**2 * out_channels, out_size**2 * out_channels),
                nn.LeakyReLU(),
                nn.Dropout(p = self.drop_p),
                nn.Linear(out_size**2 * out_channels, out_size**2 * out_channels),
                nn.LeakyReLU(),
                nn.Dropout(p = self.drop_p),
        )

        transpose_layers = []
        transpose_channels = [*reversed(channels[1:]), 1]
        for i, (out_channels, kernel_size) in enumerate(zip(transpose_channels, reversed(kernels))):
            transpose_layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                                       kernel_size = kernel_size, stride = 1, padding = 0))
            if i < len(transpose_channels) - 1:
                transpose_layers.append(nn.LeakyReLU())
            if self.drop_p > 0:
                transpose_layers.append(nn.Dropout(p = self.drop_p))
            in_channels = out_channels

        self.invconvnet = nn.Sequential(*transpose_layers)

    def get_fields(self, epsilons):
        batch_size, W, H = epsilons.shape
        out = epsilons.view(batch_size, 1, W, H)

        out = self.convnet(out)
        _, c, w2, h2 = out.shape

        out = out.view(batch_size, -1)
        out = self.densenet(out)

        out = out.view(batch_size, c, w2, h2)
        out = self.invconvnet(out)

        out = out.view(batch_size, W, H)

        return out

    def forward(self, epsilons, trim_buffer = True):
        # Compute Ez fields
        fields = self.get_fields(epsilons)

        # Compute Maxwell operator on fields
        residuals = maxwell_residual_2d(fields, epsilons, self.curl_curl_op,
                                        buffer_length = self.buffer_length, trim_buffer = trim_buffer)

        # Compute free-current vector
        if trim_buffer:
            J = torch.zeros(self.size, self.size, device = device)
            J[self.src_y, self.src_x] = -(SCALE / L0) * MU0 * OMEGA_1550
        else:
            total_size = self.size + 2 * self.buffer_length
            J = torch.zeros(total_size, total_size, device = device)
            J[self.src_y + self.buffer_length, self.src_x + self.buffer_length] = -(SCALE / L0) * MU0 * OMEGA_1550

        return residuals - J
