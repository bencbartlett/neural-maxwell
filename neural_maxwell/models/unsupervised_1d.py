import numpy as np
import torch
import torch.nn as nn

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Simulation1D, maxwell_residual
from neural_maxwell.utils import conv_output_size

class MaxwellConvV2(nn.Module):

    def __init__(self, size = DEVICE_LENGTH, src_x = 32, drop_p = 0.1, channels = None, kernels = None):
        super().__init__()

        self.size = size
        self.src_x = src_x
        self.buffer_length = 4
        self.total_size = self.size + 2 * self.buffer_length
        self.drop_p = drop_p

        curl_op, eps_op = Simulation1D(device_length = self.size, buffer_length = self.buffer_length).get_operators()
        self.curl_curl_op = torch.tensor(np.asarray(np.real(curl_op)), device = device).float()
        
        if channels is None or kernels is None:
            channels = [32, 64, 128]
            kernels = [5, 7, 9]
        
        layers = []       
        in_channels = 1
        out_size = self.size
        for out_channels, kernel_size in zip(channels, kernels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 0))
            layers.append(nn.ReLU())
            if self.drop_p > 0:
                layers.append(nn.Dropout(p = self.drop_p))
            in_channels = out_channels
            out_size = conv_output_size(out_size, kernel_size)

        self.convnet = nn.Sequential(*layers)
        
        self.densenet = nn.Sequential(
                nn.Linear(out_size * out_channels, out_size * out_channels),
                nn.ReLU(),
                nn.Dropout(p = self.drop_p),
                nn.Linear(out_size * out_channels, out_size * out_channels),
                nn.ReLU(),
                nn.Dropout(p = self.drop_p),
        )

        transpose_layers = []   
        transpose_channels = [*reversed(channels[1:]), 1]
        for i, (out_channels, kernel_size) in enumerate(zip(transpose_channels, reversed(kernels))):
            transpose_layers.append(nn.ConvTranspose1d(in_channels, out_channels, 
                                                       kernel_size = kernel_size, stride = 1, padding = 0))
            if i < len(transpose_channels) - 1:
                transpose_layers.append(nn.ReLU())
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

    def forward(self, epsilons, trim_buffer = True):
        # Compute Ez fields
        fields = self.get_fields(epsilons)

        # Compute Maxwell operator on fields
        residuals = maxwell_residual(fields, epsilons, self.curl_curl_op,
                                     buffer_length = self.buffer_length, trim_buffer = trim_buffer)

        # Compute free-current vector
        if trim_buffer:
            J = torch.zeros(self.size, 1, device = device)
            J[self.src_x, 0] = -(SCALE / L0) * MU0 * OMEGA_1550
        else:
            J = torch.zeros(self.total_size, 1, device = device)
            J[self.src_x + self.buffer_length, 0] = -(SCALE / L0) * MU0 * OMEGA_1550

        return residuals - J


class MaxwellConvComplex(nn.Module):

    def __init__(self, size = 64, src_x = None, buffer = 16, npml = 16, drop_p = 0.1, polar = False):
        super().__init__()

        self.size = size
        self.buffer = buffer
        self.npml = npml
        self.src_x = src_x if src_x is not None else self.npml + self.buffer
        self.total_size = self.npml + 2 * self.buffer + self.size + self.npml
        self.drop_p = drop_p
        self.polar = polar

        c1, c2, c3 = 32, 64, 128
        k1, k2, k3 = 3, 5, 7

        self.convnet = nn.Sequential(
                nn.Conv1d(1, c1, kernel_size = k1, stride = 1, padding = 0),
                nn.ReLU(),
                nn.Dropout(p = self.drop_p),
                nn.Conv1d(c1, c2, kernel_size = k2, stride = 1, padding = 0),
                nn.ReLU(),
                nn.Dropout(p = self.drop_p),
                nn.Conv1d(c2, c3, kernel_size = k3, stride = 1, padding = 0),
                nn.ReLU(),
                nn.Dropout(p = self.drop_p)
        )
        out_size = self.total_size - (k1 - 1) - (k2 - 1) - (k3 - 1)

        self.densenet = nn.Sequential(
                nn.Linear(out_size * c3, out_size * c3),
                nn.ReLU(),
                nn.Dropout(p = self.drop_p),
                #             nn.Linear(out_size * c3, out_size * c3),
                #             nn.ReLU(),
                #             nn.Dropout(p=self.drop_p),
        )

        self.invconvnet = nn.Sequential(
                nn.ConvTranspose1d(c3, c2, kernel_size = k3, stride = 1, padding = 0),
                nn.ReLU(),
                nn.Dropout(p = self.drop_p),
                nn.ConvTranspose1d(c2, c1, kernel_size = k2, stride = 1, padding = 0),
                nn.ReLU(),
                nn.ConvTranspose1d(c1, 2, kernel_size = k1, stride = 1, padding = 0),
        )

        curl_op, _ = Simulation1D(device_length = self.size, npml = self.npml).get_operators()
        self.curl_curl_re = torch.tensor(np.asarray(np.real(curl_op)), device = device).float()
        self.curl_curl_im = torch.tensor(np.asarray(np.imag(curl_op)), device = device).float()

    def forward_fields(self, epsilons):
        batch_size, L = epsilons.shape
        out = epsilons.view(batch_size, 1, L)

        out = self.convnet(out)

        _, c, l2 = out.shape
        out = out.view(batch_size, -1)
        out = self.densenet(out)
        out = out.view(batch_size, c, l2)

        out = self.invconvnet(out)

        out = out.view(batch_size, 2, L)

        return out

    def get_fields(self, epsilons):
        # Compute Ez fields
        fields = self.forward_fields(epsilons)

        # Separate into real and imaginary parts
        if not self.polar:
            E_re = fields[:, 0]
            E_im = fields[:, 1]
        else:
            E_abs = fields[:, 0]
            E_phi = fields[:, 1]
            E_re = E_abs * torch.cos(E_phi)
            E_im = E_abs * torch.sin(E_phi)

        return E_re, E_im

    def forward(self, epsilons):

        batch_size, _ = epsilons.shape

        E_re, E_im = self.get_fields(epsilons)

        # Broadcast E and epsilon vectors for matrix multiplication
        E_re = E_re.view(batch_size, -1, 1)
        E_im = E_im.view(batch_size, -1, 1)
        eps = epsilons.view(batch_size, -1, 1)

        # Compute Maxwell operator on fields
        curl_curl_E_re = (SCALE / L0 ** 2) * (torch.matmul(self.curl_curl_re, E_re)
                                              - torch.matmul(self.curl_curl_im, E_im))
        curl_curl_E_im = (SCALE / L0 ** 2) * (torch.matmul(self.curl_curl_im, E_re)
                                              + torch.matmul(self.curl_curl_re, E_im))

        epsilon_E_re = (SCALE * -OMEGA_1550 ** 2 * MU0 * EPSILON0) * eps * E_re
        epsilon_E_im = (SCALE * -OMEGA_1550 ** 2 * MU0 * EPSILON0) * eps * E_im

        # Compute free-current vector
        J_re = torch.zeros(batch_size, self.total_size, 1, device = device)
        J_im = torch.zeros(batch_size, self.total_size, 1, device = device)
        J_re[:, self.src_x, 0] = -1.526814027933079  # source is in phase with real part

        out_re = curl_curl_E_re - epsilon_E_re - J_re
        out_im = curl_curl_E_im - epsilon_E_im - J_im

        CONCAT = True
        if CONCAT:
            # output is concatenated real and imaginary part
            out = torch.cat((out_re, out_im), dim = -1)
        else:
            # output is sum of real and imaginary part
            out = torch.abs(out_re) + torch.abs(out_im)

        return out
