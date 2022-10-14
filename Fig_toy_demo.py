import sys

sys.path.append('../experiments/regression/')

from copy import deepcopy
import math
import numpy as np
import torch
import pandas as pd
import gpytorch

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2.5)

from gpytorch import mlls
from online_gp import models
from online_gp.models.stems import Identity
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence as KL_div

sns.set_style('white')




def draw_plot(model, train_x, train_y, test_x, test_y, inducing_x, inducing_f, ax, color, show_legend=False):
    inducing_x = inducing_x.detach().cpu()
    inducing_f = inducing_f.detach().cpu()
    train_x, train_y = train_x.cpu().squeeze(-1), train_y.cpu().squeeze(-1)
    test_x, test_y = test_x.cpu().squeeze(-1), test_y.cpu().squeeze(-1)

    x_min = min(train_x.min(), test_x.min(), inducing_x.min())
    x_max = max(train_x.max(), test_x.max(), inducing_x.max())
    xlim = (x_min - 1e-1, x_max + 1e-1)
    x_grid = torch.linspace(*xlim, 200)

    model.eval()
    with torch.no_grad():
        mean, var = model.predict(x_grid)
        #         pred_dist = model(x_grid)
        #         mean, var = pred_dist.mean, pred_dist.variance
        lb, ub = mean - 2 * var.sqrt(), mean + 2 * var.sqrt()

    mean = mean.cpu().view(-1)
    lb, ub = lb.cpu().view(-1), ub.cpu().view(-1)

    ax.plot(x_grid.cpu(), mean, linewidth=1, color=color)
    ax.fill_between(x_grid.cpu(), lb, ub, alpha=0.22, color=color)

    ax.scatter(train_x.cpu(), train_y.cpu(), color="black", s=32, edgecolors='none', label='observed')
    if test_x is not None:
        ax.scatter(test_x.cpu(), test_y.cpu(), color="black", s=32, facecolors='none', label='unobserved')

    ax.scatter(inducing_x, inducing_f, color="red", marker="+", linewidth=3, s=128, label='inducing')

    ax.set_xlim((-1.1, 1.1))
    ax.set_xlabel('x')
    ax.set_ylim((-3, 3))
    sns.despine()
    if show_legend:
        ax.legend(ncol=3, bbox_to_anchor=(0., 1.2))
    plt.tight_layout()
    return ax





