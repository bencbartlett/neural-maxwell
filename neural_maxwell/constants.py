from numpy import pi
from torch import device as torch_device

N_si = 3.48
eps_si = N_si**2

N_sio2 = 1.44
eps_sio2 = N_sio2**2

N_sinitride = 1.99
eps_sinitride = N_sinitride**2

OMEGA_1550 = 1.215e15
GRID_SIZE = 64
NPML = 15
OMEGA = OMEGA_1550
MU0 = 4 * pi * 10**-7
EPSILON0 =  8.854187817620e-12
SCALE = 1e-15
C = 299792458.0
L0 = 1e-6
PIXEL_SIZE = 0.05 * L0

# Device configuration
device = torch_device('cuda:0')
# device = torch_device('cpu')