from numpy import pi
from torch import device as torch_device

# Refractive indices and permittivities ================================================================================

N_si = 3.48  # silicon
eps_si = N_si ** 2

N_sio2 = 1.44  # silicon oxide
eps_sio2 = N_sio2 ** 2

N_sinitride = 1.99  # silicon nitride
eps_sinitride = N_sinitride ** 2

# Physical constants ===================================================================================================
OMEGA_1550 = 1.215e15  # 1550nm frequency
MU0 = 4 * pi * 10 ** -7  # vacuum permeability
EPSILON0 = 8.854187817620e-12  # vacuum permittivity
C = 299792458.0  # speed of light
BUFFER_PERMITTIVITY = -1e20  # near infinite permittivity for cavity boundaries
# BUFFER_PERMITTIVITY = 1.0  # vacuum permittivity if using PML

# Design space size ====================================================================================================
DEVICE_LENGTH = 64  # length of permittivity region
DEVICE_LENGTH_2D = 32
NPML = 0  # number of PMLs
BUFFER_LENGTH = 4  # buffer size before NPML (reflective boundary if using cavity)
SCALE = 1e-15
L0 = 1e-6
dL = 0.05
PIXEL_SIZE = dL * L0

# Device configuration =================================================================================================
device = torch_device('cuda:0')
# device = torch_device('cpu')
