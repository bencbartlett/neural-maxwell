import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn

from time import time

from neural_maxwell.datasets.fdfd import Simulation1D
from neural_maxwell.constants import *

def plot_model_outputs(model, epsilons, figsize=(18,12), criterion=nn.MSELoss(), rescale=True):
    
    size = model.size
    buffer_length = model.buffer_length
    
    start = time()
    _, src_x, Hx, Hy, Ez = Simulation1D(device_length=size, 
                                        buffer_length=buffer_length).solve(epsilons, omega=OMEGA_1550)
    sim_time = time() - start
    
    Ez_true = np.real(Ez)[buffer_length:-buffer_length]

    eps_tensor = torch.tensor([epsilons], device=device).float()
    start = time()
    fields = model.get_fields(eps_tensor)
    network_time = time() - start
    
    Ez_pred = fields[0].detach().cpu().numpy()
    if rescale:
        scale_ratio = np.mean(np.abs(Ez_true)) / np.mean(np.abs(Ez_pred))
        Ez_pred *= scale_ratio
    
    outputs = model(eps_tensor)
    
    loss = criterion(outputs, torch.zeros_like(outputs))
    outputs = outputs[0].detach().cpu().numpy() 
    
    print(f"Error: {np.linalg.norm(outputs)}")
    print(f"Loss: {loss.item()}")

    print("Sim time: {:.5f} | Network time: {:.5f} | Ratio: {:.5f}".format(
        sim_time, network_time, network_time / sim_time))

    # Make initial figure
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=figsize, gridspec_kw = {'height_ratios':[3, 1]})
    fig.subplots_adjust(hspace=0)
    for a in ax:
        a.label_outer()
    top, bot = ax

    x = PIXEL_SIZE / L0 * (np.arange(len(Ez_true)) - src_x)
    top.plot(x, epsilons, label="$\epsilon (x)$", c='grey')
    top.plot(x, np.zeros(outputs.shape), linestyle=':', c='grey')
    top.fill_between(x, epsilons, alpha=0.1, color='grey')
    top.plot(x, Ez_true, label="$E_\mathrm{true}$", c='C0')
    Ez_pred_line = top.plot(x, Ez_pred, label="$E_\mathrm{pred}$", c='C1')
    top.legend(loc="lower right", prop={"size": 16})

    outputs_line = bot.plot(x, outputs, label="$\\nabla \\times \\nabla \\times E - \omega^2 \mu_0 \epsilon E - J$", c='red')
    bot.fill_between(x, outputs.flatten(), alpha=0.05, color='red')
    bot.plot(x, np.zeros(outputs.shape), linestyle=':', c='grey')
    bot.set_ylim(-.25, .25)
    bot.set_xlabel("$x (\mu m)$")
    bot.text(0.05, 0.85, '$\mathcal{{L}}=${:.1e}'.format(loss.item()), fontsize='xx-large', transform=ax[1].transAxes)
    bot.legend(loc="upper right", prop={"size": 16})

    plt.show()