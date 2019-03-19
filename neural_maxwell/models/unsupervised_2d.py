import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Cavity2D


class MaxwellConv2D(nn.Module):

    def __init__(self, size = 32, src_x = 16, batch_size = 50, supervised = False):
        super().__init__()

        self.size = size
        self.src_x = src_x
        self.supervised = supervised
        self.cavity_buffer = 4
        self.total_size = self.size + 2 * self.cavity_buffer

        self.convnet = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 0),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 0),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size = 7, stride = 1, padding = 0),
                nn.ReLU()
        )
        out_size = size - 2 - 4 - 6

        self.densenet = nn.Sequential(
                nn.Linear(out_size ** 2 * 32, out_size ** 2 * 32),
                nn.ReLU(),
                nn.Linear(out_size ** 2 * 32, out_size ** 2 * 32),
                nn.ReLU(),
        )

        self.invconvnet = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size = 7, stride = 1, padding = 0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size = 5, stride = 1, padding = 0),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size = 3, stride = 1, padding = 0),
        )

        # store angler operators
        curl_op, eps_op = Cavity2D(device_length = self.size, cavity_buffer = self.cavity_buffer).get_operators()
        self.curl_curl_op = torch.tensor([np.asarray(np.real(curl_op))] * batch_size, device = device).float()

    @staticmethod
    def make_convolutional_mininet(num_input_channels, channels, kernel_sizes):
        layers = []
        in_channels = num_input_channels
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 0))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

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

    def forward(self, epsilons):
        # Compute Ez fields
        fields = self.get_fields(epsilons)

        if self.supervised:
            labels = torch.empty_like(fields)
            for i, perm in enumerate(epsilons.detach().numpy()):
                _, _, _, _, Ez = Cavity2D(cavity_buffer = 16).solve(perm, omega = OMEGA_1550)
                labels[i, :] = torch.tensor(np.real(Ez[16:-16])).float()
            return fields - labels

        else:
            batch_size, _, _ = epsilons.shape

            # Add zero field amplitudes at edge points for resonator BC's
            E = F.pad(fields, [self.cavity_buffer] * 4)
            E = E.view(batch_size, -1, 1)

            # Add first layer of cavity BC's
            eps = F.pad(epsilons, [self.cavity_buffer] * 4, "constant", -1e20)
            eps = eps.view(batch_size, -1, 1)

            # Compute Maxwell operator on fields
            curl_curl_E = (SCALE / L0 ** 2) * torch.bmm(self.curl_curl_op, E).view(batch_size, -1, 1)
            epsilon_E = (SCALE * -OMEGA ** 2 * MU0 * EPSILON0) * eps * E

            # Compute free-current vector
            J = torch.zeros(batch_size, self.total_size, self.total_size, device = device)
            J[:, self.src_x + self.cavity_buffer, self.src_x + self.cavity_buffer] = -1.526814027933079
            J = J.view(batch_size, -1, 1)

            out = curl_curl_E - epsilon_E - J
            out = out.view(batch_size, self.total_size, self.total_size)
            out = out[:, self.cavity_buffer:-self.cavity_buffer, self.cavity_buffer:-self.cavity_buffer]

            return out
