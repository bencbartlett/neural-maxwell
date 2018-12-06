from typing import Optional, Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook


def is_notebook():
    '''Tests to see if we are running in a jupyter notebook environment'''
    try:
        # noinspection PyUnresolvedReferences
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

# def tqdm_train(epochs: int = None, data: Iterable = None, plot=is_notebook()):
#     epoch_bar = pbar(range(epochs))
    

# class TqdmTrainer:
#     '''Progress bar for displaying training progress and stats'''
#     def __init__(self, epochs: int = None, data: Iterable = None, plot=is_notebook()):
#         self.epoch_bar = pbar(range(epochs), desc="ℒ = {:.1e}")
#         self.iteration_bar = pbar(data)
    
#     def __iter__(self):
#         pass


def tensor_diff(x, n=1, axis=-1, padding=None, pad_value=0, cyclic=False):
    '''PyTorch equivalent of numpy.diff()'''
    if n == 0:
        return x
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))

    # Pad tensor if needed
    if padding is True:
        dims = 2 * len(x.shape)
        pad = [0] * dims
        pad[2 * axis] = 1
        # pad[2 * axis + 1] = 1 # don't pad right since diffs are left aligned
        pad = tuple(reversed(pad))
        a = F.pad(x, pad, mode="constant", value=pad_value)
    elif padding is not None:
        a = F.pad(x, padding, mode="constant", value=pad_value)
    else:
        a = x

    if cyclic:
        # TODO: not complete
        a0 = tensor_roll(a, 0, axis=axis)
        a1 = tensor_roll(a, -1, axis=axis)
    else:
        length = a.shape[axis] - 1
        a0 = a.narrow(axis, 0, length)
        a1 = a.narrow(axis, 1, length)
    return tensor_diff(a1 - a0, n=(n - 1), axis=axis, padding=padding, pad_value=pad_value, cyclic=cyclic)


def tensor_roll(x: torch.Tensor, shift: int, axis: int = -1, fill_pad: Optional[int] = None):
    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(axis, torch.arange(shift))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(axis, torch.arange(shift, x.size(axis))), gap], dim=axis)

    else:
        shift = x.size(axis) - shift
        gap = x.index_select(axis, torch.arange(shift, x.size(axis)))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(axis, torch.arange(shift))], dim=axis)
