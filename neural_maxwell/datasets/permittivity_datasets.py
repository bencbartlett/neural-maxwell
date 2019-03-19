import numpy as np
import torch
from torch.utils.data import Dataset

from neural_maxwell.datasets.generators import load_batch


class Permittivity1dSupervisedDataset(Dataset):

    def __init__(self, hdf5_file, batch_name, kernel_sizes = []):
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
        fields = torch.tensor(np.real(self.Ez[i]))
        return epsilons.float(), src_x, fields.float()


class PermittivityDataset(Dataset):

    def __init__(self, epsilon_generator, N = 10000, size = 64, infinite_mode = False):
        self.size = size
        self.epsilon_generator = epsilon_generator
        self.epsilon_samples = []
        self.N = N
        self.infinite_mode = infinite_mode

    def __len__(self):
        return int(self.N)

    def __getitem__(self, i):
        if i >= len(self.epsilon_samples) or self.infinite_mode:
            epsilons = torch.tensor(self.epsilon_generator())
            if not self.infinite_mode:
                self.epsilon_samples.append(epsilons)
            return epsilons
        else:
            epsilons = self.epsilon_samples[i]
            return epsilons


# class PermittivityLayerDataset(Dataset):
#
#     def __init__(self, layer_generator, N = 10000, size = 64, infinite_mode = False):
#         self.size = size
#         self.layer_generator = layer_generator
#         self.layer_samples = []
#         self.N = N
#         self.infinite_mode = infinite_mode
#
#     def __len__(self):
#         return int(self.N)
#
#     def __getitem__(self, i):
#         if i >= len(self.layer_samples) or self.infinite_mode:
#             sizes, epsilons = self.layer_generator()
#             sizes = torch.tensor(sizes)
#             epsilons = torch.tensor(epsilons)
#             if not self.infinite_mode:
#                 self.layer_samples.append((sizes, epsilons))
#             return sizes, epsilons
#         else:
#             sizes, epsilons = self.layer_samples[i]
#             return sizes, epsilons
