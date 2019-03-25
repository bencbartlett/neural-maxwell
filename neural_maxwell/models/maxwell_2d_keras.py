import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Dense

from neural_maxwell.constants import *
from neural_maxwell.datasets.fdfd import Simulation2D, maxwell_residual_2d_tf
from neural_maxwell.utils import conv_output_size


class MaxwellSolverPML2D(tf.keras.Model):

    def __init__(self, size = DEVICE_LENGTH_2D, src_x = 16, src_y = 16, buffer_length = 4, npml = 8,
                 channels = None, kernels = None, drop_p = 0.1):
        super(MaxwellSolverPML2D, self).__init__()

        self.size = size
        self.src_x = src_x
        self.src_y = src_y
        self.buffer_length = buffer_length
        self.npml = npml
        self.drop_p = drop_p

        self.sim = Simulation2D(device_length = self.size, npml = self.npml, buffer_length = self.buffer_length)
        curl_curl_op, eps_op = self.sim.get_operators()
        self.curl_curl_op = tf.convert_to_tensor(curl_curl_op, dtype = tf.complex64)

        if channels is None or kernels is None:
            channels = [64] * 7
            kernels = [5] * 7

        layers = []
        out_size = self.size
        for out_channels, kernel_size in zip(channels, kernels):
            layers.append(Conv2D(out_channels, kernel_size = kernel_size, padding = 'valid'))
            layers.append(LeakyReLU())
            if self.drop_p > 0:
                layers.append(Dropout(self.drop_p))
            out_size = conv_output_size(out_size, kernel_size)

        self.convnet = tf.keras.Sequential(layers)

        self.densenet = tf.keras.Sequential([
            Dense(out_size ** 2 * out_channels),
            LeakyReLU(),
            # Dropout(self.drop_p),
            Dense(out_size ** 2 * out_channels),
            LeakyReLU(),
            # Dropout(self.drop_p),
        ])

        transpose_layers = []
        transpose_channels = [*reversed(channels[1:]), 1]
        for i, (out_channels, kernel_size) in enumerate(zip(transpose_channels, reversed(kernels))):
            transpose_layers.append(Conv2DTranspose(out_channels, kernel_size = kernel_size, stride = 1, padding = 0))
            if i < len(transpose_channels) - 1:
                transpose_layers.append(LeakyReLU())
            if self.drop_p > 0:
                transpose_layers.append(Dropout(self.drop_p))

        self.invconvnet = tf.keras.Sequential(transpose_layers)

    def get_fields(self, epsilons):
        batch_size, W, H = epsilons.shape
        out = tf.reshape(epsilons, (batch_size, 1, W, H))

        out = self.convnet(out)
        _, c, w2, h2 = out.shape

        out = tf.reshape(out, (batch_size, -1))
        out = self.densenet(out)

        out = tf.reshape(out, (batch_size, c, w2, h2))
        out = self.invconvnet(out)

        out = tf.reshape(out, (batch_size, W, H))

        return out

    def call(self, epsilons, trim_buffer = False):

        # Compute Ez fields
        fields = self.get_fields(epsilons)

        # Compute Maxwell operator on fields
        residuals = maxwell_residual_2d_tf(fields, epsilons, self.curl_curl_op,
                                           buffer_length = self.buffer_length, trim_buffer = trim_buffer)

        # Compute free-current vector
        if trim_buffer:
            J = tf.zeros((self.size, self.size), dtype = tf.complex128)
            J[self.src_y, self.src_x] = -(SCALE / L0) * MU0 * OMEGA_1550
        else:
            total_size = self.size + 2 * self.buffer_length
            J = tf.zeros((total_size, total_size), dtype = tf.complex128)
            J[self.src_y + self.buffer_length, self.src_x + self.buffer_length] = -(SCALE / L0) * MU0 * OMEGA_1550

        return residuals - J
