import numpy as np
import torch
import torch.nn as nn

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Simulation2D, maxwell_residual_2d, maxwell_residual_2d_complex
from neural_maxwell.utils import conv_output_size


class MaxwellSolver2D(nn.Module):

    def __init__(self, size = 32, buffer_length = 4, buffer_permittivity = BUFFER_PERMITTIVITY, npml = 0, src_x = 16,
                 src_y = 16, channels = None, kernels = None, drop_p = 0.1):

        super().__init__()

        self.size = size
        self.src_x = src_x
        self.src_y = src_y
        self.npml = npml
        self.use_complex = (self.npml == 0)
        self.buffer_length = buffer_length
        self.buffer_permittivity = buffer_permittivity
        self.drop_p = drop_p

        self.sim = Simulation2D(device_length = self.size, buffer_length = self.buffer_length, npml = self.npml,
                                buffer_permittivity = self.buffer_permittivity)
        curl_curl_op, eps_op = self.sim.get_operators()
        if self.use_complex:
            self.curl_curl_op = torch.tensor(np.asarray(np.real(curl_curl_op)), device = device).float()
        else:
            self.curl_curl_re = torch.tensor(np.asarray(np.real(curl_curl_op)), device = device).float()
            self.curl_curl_im = torch.tensor(np.asarray(np.imag(curl_curl_op)), device = device).float()

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
                nn.Linear(out_size ** 2 * out_channels, out_size ** 2 * out_channels),
                nn.LeakyReLU(),
                nn.Dropout(p = self.drop_p),
                nn.Linear(out_size ** 2 * out_channels, out_size ** 2 * out_channels),
                nn.LeakyReLU(),
                nn.Dropout(p = self.drop_p),
        )

        transpose_layers = []
        transpose_channels = [*reversed(channels[1:]), 1 if self.use_complex else 2]
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

        if self.use_complex:
            out = out.view(batch_size, 2, W, H)
            return out[:, 0], out[:, 1]
        else:
            out = out.view(batch_size, W, H)
            return out

    def forward_real(self, fields, epsilons, trim_buffer = True):

        # Compute Maxwell operator on fields
        residuals = maxwell_residual_2d(fields, epsilons, self.curl_curl_op,
                                        buffer_length = self.buffer_length, trim_buffer = trim_buffer)

        if trim_buffer:
            J = torch.zeros(self.size, self.size, device = device)
            J[self.src_y, self.src_x] = -(SCALE / L0) * MU0 * OMEGA_1550
        else:
            total_size = self.size + 2 * self.buffer_length
            J = torch.zeros(total_size, total_size, device = device)
            J[self.src_y + self.buffer_length, self.src_x + self.buffer_length] = -(SCALE / L0) * MU0 * OMEGA_1550

        return residuals - J

    def forward_complex(self, fields_re, fields_im, epsilons, trim_buffer = False):
        # Compute Maxwell operator on fields
        res_re, res_im = maxwell_residual_2d_complex(fields_re, fields_im, epsilons,
                                                     self.curl_curl_re, self.curl_curl_im,
                                                     buffer_length = self.buffer_length,
                                                     add_buffer = False, trim_buffer = False)
        if trim_buffer:
            J_re = torch.zeros(self.size, self.size, device = device)
            J_im = torch.zeros(self.size, self.size, device = device)
            J_im[self.src_y, self.src_x] = -(SCALE / L0) * MU0 * OMEGA_1550
        else:
            total_size = self.size + 2 * self.buffer_length
            J_re = torch.zeros(total_size, total_size, device = device)
            J_im = torch.zeros(total_size, total_size, device = device)
            J_im[self.src_y + self.buffer_length, self.src_x + self.buffer_length] = -(
                    SCALE / L0) * MU0 * OMEGA_1550

            out_re = res_re - J_re
            out_im = res_im - J_im

            CONCAT = True
            if CONCAT:
                # output is concatenated real and imaginary part
                out = torch.cat((out_re, out_im), dim = -1)
            else:
                # output is sum of real and imaginary part
                out = torch.sqrt(out_re ** 2 + out_im ** 2)

    def forward(self, epsilons, trim_buffer = True):
        # Compute Ez fields
        if self.use_complex:
            fields_re, fields_im = self.get_fields(epsilons)
            return self.forward_complex(fields_re, fields_im, epsilons)
        else:
            fields = self.get_fields(epsilons)
            return self.forward_real(fields, epsilons)
