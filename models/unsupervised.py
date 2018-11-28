import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets.generators1d import *
from .utils import tensor_diff

class Perm1dDataset_test(Dataset):

    def __init__(self, hdf5_file, batch_name, kernel_sizes=[]):
        data = load_batch(hdf5_file, batch_name)

        self.epsilons = data["epsilons"]
        self.src = data["src"]
        self.Ez = data["Ez"]

        self.input_size = self.Ez[0].shape[0]
        output_size = self.input_size
        for kernel_size in kernel_sizes:
            stride = 1
            output_size = (output_size - kernel_size) / stride + 1
        self.output_size = int(output_size)

    def __len__(self):
        return len(self.epsilons)

    def __getitem__(self, i):
        epsilons = torch.tensor(self.epsilons[i])
        src_x = self.src[i]
        fields = torch.tensor(np.imag(self.Ez[i]))  # TODO: change to real
        return epsilons.float(), src_x, fields.float()


class Perm1dDataset(Dataset):

    def __init__(self, epsilon_generator, kernel_sizes=[], N=10000, input_size=128, infinite_mode=False):
        self.input_size = input_size
        output_size = self.input_size
        for kernel_size in kernel_sizes:
            stride = 1
            output_size = (output_size - kernel_size) / stride + 1
        self.output_size = int(output_size)

        self.epsilon_generator = epsilon_generator
        self.epsilon_samples = []
        self.src_samples = []
        self.N = N
        self.infinite_mode = infinite_mode

    def __len__(self):
        if self.infinite_mode:
            return int(1e8)
        else:
            return int(self.N)

    def __getitem__(self, i):
        if i >= len(self.epsilon_samples) or self.infinite_mode:
            #             src = torch.zeros(self.input_size)
            #             src_x = np.random.randint(1,32)
            #             src[src_x] = 1.0
            epsilons = np.ones(self.input_size)
            epsilons[32:96] = self.epsilon_generator()
            epsilons = torch.tensor(epsilons).float()
            src_x = np.random.randint(1, 127 - 1)  # 16 # TODO
            if not self.infinite_mode:
                self.epsilon_samples.append(epsilons)
                self.src_samples.append(src_x)
            return epsilons, src_x, torch.zeros(epsilons.shape).float()
        else:
            epsilons = self.epsilon_samples[i]
            src_x = self.src_samples[i]
            return epsilons.float(), src_x, torch.zeros(epsilons.shape).float()

OMEGA = 1.215e15
MU0 = 4 * np.pi * 10**-7
EPSILON0 =  8.854187817620e-12
SCALE = 1e-15
C = 299792458.0
L0 = 1e-6
PIXEL_SIZE = 0.05 * L0
wavelength = 2 * np.pi * C / OMEGA

class MaxwellDense(nn.Module):
    def __init__(self, size=128):
        super().__init__()

        self.size = size

        #         self.layer_dims = [self.size, 256, 512, 512, 256, self.size]
        self.layer_dims = [2 * self.size, 512, 256, self.size]

        layers_amp = []
        layers_phi = []
        for i in range(len(self.layer_dims) - 1):
            layers_amp.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            layers_phi.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))

        self.layers_amp = nn.ModuleList(layers_amp)
        self.layers_phi = nn.ModuleList(layers_phi)

    def forward_amplitude_phase(self, x):
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

    def get_fields(self, epsilons, src):
        # Get amplitude and phase vectors
        data = torch.cat((epsilons, src), dim=-1)
        A, phi = self.forward_amplitude_phase(data)  # epsilons)

        # Combine to form waveform
        x = (PIXEL_SIZE * (torch.arange(self.size)  # - src_x
                           )).float()
        fields = A * torch.cos(OMEGA / C * torch.sqrt(epsilons) * x + phi)
        return fields

    def forward(self, epsilons, src):

        fields = self.get_fields(epsilons, src)

        eps = epsilons[:, 1:-1]
        E = fields[:, 1:-1]

        # Compute Maxwell operator on fields
        curl_curl_E = tensor_diff(fields, n=2)
        left_factor = SCALE / PIXEL_SIZE ** 2
        right_factor = SCALE * -OMEGA ** 2 * MU0 * EPSILON0

        out = (left_factor * curl_curl_E) - (right_factor * eps * E) - src[:, 1:-1]

        return out
