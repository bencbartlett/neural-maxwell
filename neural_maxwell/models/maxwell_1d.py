import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Simulation1D, maxwell_residual_1d
from neural_maxwell.utils import conv_output_size


class MaxwellSolver1D(nn.Module):

    def __init__(self, size = DEVICE_LENGTH, src_x = 32, channels = None, kernels = None, drop_p = 0.1):
        super().__init__()

        self.size = size
        self.src_x = src_x
        self.buffer_length = 4
        self.drop_p = drop_p

        self.sim = Simulation1D(device_length = self.size, buffer_length = self.buffer_length)
        curl_op, eps_op = self.sim.get_operators()
        self.curl_curl_op = torch.tensor(np.asarray(np.real(curl_op)), device = device).float()

        if channels is None or kernels is None:
            channels = [64] * 7
            kernels = [5] * 7

        layers = []
        in_channels = 1
        out_size = self.size
        for out_channels, kernel_size in zip(channels, kernels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 0))
            layers.append(nn.LeakyReLU())
            if self.drop_p > 0:
                layers.append(nn.Dropout(p = self.drop_p))
            in_channels = out_channels
            out_size = conv_output_size(out_size, kernel_size)

        self.convnet = nn.Sequential(*layers)

        self.densenet = nn.Sequential(
                nn.Linear(out_size * out_channels, out_size * out_channels),
                nn.LeakyReLU(),
                nn.Dropout(p = self.drop_p),
                nn.Linear(out_size * out_channels, out_size * out_channels),
                nn.LeakyReLU(),
                nn.Dropout(p = self.drop_p),
        )

        transpose_layers = []
        transpose_channels = [*reversed(channels[1:]), 1]
        for i, (out_channels, kernel_size) in enumerate(zip(transpose_channels, reversed(kernels))):
            transpose_layers.append(nn.ConvTranspose1d(in_channels, out_channels,
                                                       kernel_size = kernel_size, stride = 1, padding = 0))
            if i < len(transpose_channels) - 1:
                transpose_layers.append(nn.LeakyReLU())
            if self.drop_p > 0:
                transpose_layers.append(nn.Dropout(p = self.drop_p))
            in_channels = out_channels

        self.invconvnet = nn.Sequential(*transpose_layers)

    def get_fields(self, epsilons):
        batch_size, L = epsilons.shape
        out = epsilons.view(batch_size, 1, L)

        out = self.convnet(out)
        _, c, l2 = out.shape

        out = out.view(batch_size, -1)
        out = self.densenet(out)

        out = out.view(batch_size, c, l2)
        out = self.invconvnet(out)

        out = out.view(batch_size, L)

        return out

    def forward_unsupervised(self, epsilons, fields, trim_buffer = True):

        # Compute Maxwell operator on fields
        residuals = maxwell_residual_1d(fields, epsilons, self.curl_curl_op,
                                        buffer_length = self.buffer_length, trim_buffer = trim_buffer)

        # Compute free-current vector
        if trim_buffer:
            J = torch.zeros(self.size, 1, device = device)
            J[self.src_x, 0] = -(SCALE / L0) * MU0 * OMEGA_1550
        else:
            J = torch.zeros(self.size + 2 * self.buffer_length, 1, device = device)
            J[self.src_x + self.buffer_length, 0] = -(SCALE / L0) * MU0 * OMEGA_1550

        return residuals - J

    def forward_supervised(self, epsilons, fields, trim_buffer = True):

        batch_size, _ = epsilons.shape

        fields_true = []
        epsilons_np = epsilons.detach().cpu().numpy()
        for eps_np in epsilons_np:
            _, _, _, _, Ez_true = self.sim.solve(eps_np, src_x = self.src_x)
            fields_true.append(np.real(Ez_true))
        fields_true = np.array(fields_true)

        if trim_buffer:
            fields_true = fields_true[:, self.buffer_length: -self.buffer_length]
        else:
            fields = F.pad(fields, [self.buffer_length] * 2)

        fields_true = torch.tensor(fields_true, device = device).float()

        return fields_true - fields

    def forward(self, epsilons, supervised = False, trim_buffer = True):
        # Compute Ez fields
        fields = self.get_fields(epsilons)

        if supervised:
            return self.forward_supervised(epsilons, fields, trim_buffer = trim_buffer)
        else:
            return self.forward_unsupervised(epsilons, fields, trim_buffer = trim_buffer)
