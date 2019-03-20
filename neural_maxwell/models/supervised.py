import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from neural_maxwell.datasets.generators import load_batch

class Perm1dDataset(Dataset):
    """Dataset for 1D permittivity/field data"""

    def __init__(self, hdf5_file, batch_name, kernel_sizes):
        data = load_batch(hdf5_file, batch_name)

        self.epsilons = data["epsilons"]
        self.Hx = data["Hx"]
        self.Hy = data["Hy"]
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
        data = torch.tensor([self.epsilons[i]]).float()

        # start = (self.input_size - self.output_size) // 2
        # end = start + self.output_size
        # label = torch.tensor(np.abs(self.Ez[i][start:end, start:end] * self.field_scale)).float()

        label = torch.tensor([np.abs(self.Ez[i][-1])]).float()

        return data, label


class Conv1dNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_channels = 1
        self.output_channels = 1

        self.conv_channels = [16, 16, 24, 32, 24]
        self.kernel_sizes = [25, 11, 7, 5, 3]

        self.input_size = 96
        output_size = self.input_size
        for kernel_size in self.kernel_sizes:
            stride = 1
            output_size = (output_size - kernel_size) / stride + 1
        self.output_size = int(output_size)

        layers = []

        in_channels = self.input_channels  # number of input channels
        for out_channels, kernel_size in zip(self.conv_channels, self.kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0))
            in_channels = out_channels

        self.convolutions = nn.ModuleList(layers)
        self.conv_output_size = self.output_size * self.conv_channels[-1]

        self.hidden_size = 256
        self.dense1 = nn.Linear(self.conv_output_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.convolutions):
            out = layer(out)
            out = nn.ReLU()(out)

        # Flatten
        out = torch.reshape(out, (-1, self.conv_output_size))

        # Dense layer
        out = self.dense1(out)
        out = nn.ReLU()(out)
        out = self.dense2(out)

        return out
