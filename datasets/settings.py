from tqdm import tqdm, tqdm_notebook

N_si = 3.48
eps_si = N_si**2

N_sio2 = 1.44
eps_sio2 = N_sio2**2

N_sinitride = 1.99
eps_sinitride = N_sinitride**2

OMEGA_1550 = 1.215e15

GRID_SIZE = 64
NPML = 15


def is_notebook():
    '''Tests to see if we are running in a jupyter notebook environment'''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


pbar = tqdm_notebook if is_notebook() else tqdm
