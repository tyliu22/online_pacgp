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
# torch.__version__


"""
Function: online gp pac model for sin/cos/fx dataset
        
# Dataset: fn = 'fx' OR 'sin' OR 'cos'
# IID data: shuffle=False; Non IID data: shuffle=True
X, Y = get_data(num_rows=4 * chunk_size - 1, shuffle=False, fn='cos')    
    
"""

def random_split(*tensors, sizes):
    num_rows = tensors[0].size(0)
    assert len(sizes) == 2
    assert sum(sizes) == num_rows

    a_idxs = torch.randint(0, num_rows, (sizes[0],))
    a_mask = torch.zeros(num_rows).bool()
    a_mask[a_idxs] += 1
    b_idxs = torch.masked_select(torch.arange(0, num_rows), ~a_mask)

    split_tensors = [(tensor[a_idxs], tensor[b_idxs]) for tensor in tensors]
    return split_tensors


def get_data(num_rows, shuffle=False, fn='fx'):
    if fn == 'fx':
        fx_rawdata = pd.read_csv(
            'https://raw.githubusercontent.com/trungngv/cogp/master/data/fx/fx2007-processed.csv',
            header=None
        )
        X = torch.arange(0, fx_rawdata[3].shape[0]).view(-1, 1).float()
        Y = torch.from_numpy(fx_rawdata[3].values).float()
    elif fn == 'sin':
        X = torch.linspace(-1, 1, num_rows)
        Y = torch.sin(4 * X) + 4e-1 * torch.randn(*X.shape)
    elif fn == 'cos':
        X = torch.linspace(-1, 1, num_rows)
        Y = torch.cos(4 * X) + 4e-1 * torch.randn(*X.shape)

    X, Y = X.view(-1, 1), Y.view(-1, 1)
    X, Y = X[:num_rows], Y[:num_rows]

    if shuffle:
        row_perm = torch.randperm(X.size(0))
        X = X[row_perm]
        Y = Y[row_perm]

    return X, Y


def preprocess_data(X, Y, num_init):
    dataset_size = X.size(0)

    x_min, _ = X.min(0)
    x_max, _ = X.max(0)
    x_range = x_max - x_min
    X = 2 * ((X - x_min) / x_range - 0.5)

    tmean = Y.mean()
    tstd = Y.std()
    Y = (Y - tmean) / tstd

    init_x, X = X[:num_init], X[num_init:]
    init_y, Y = Y[:num_init], Y[num_init:]

    return init_x, init_y, X, Y


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


## Plot Options

palette = sns.color_palette('bright')
model_colors = dict(exact_gp_regression=palette[7], wiski_gp_regression=palette[4], svgp_regression=palette[0],
                    sgpr=palette[9])
chunk_size = 10





# O-SGPR, time-series data
print("Start ONLINE PAC-GP, time-series data")

inducing_point_init = (-1, 1)
# Dataset: fn = 'fx' OR 'sin' OR 'cos'
# IID data: shuffle=False; Non IID data: shuffle=True
X, Y = get_data(num_rows=4 * chunk_size - 1, shuffle=False, fn='cos')
init_x, init_y, X, Y = preprocess_data(X, Y, num_init=chunk_size)

covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3)

inducing_points = torch.linspace(*inducing_point_init, 12)
osgpr_model = models.StreamingSGPR(inducing_points, learn_inducing_locations=True,
                                   covar_module=covar_module, num_data=init_x.size(0), jitter=1e-3)
osgpr_model = osgpr_model

elbo = mlls.VariationalELBO(osgpr_model.likelihood, osgpr_model, num_data=init_x.size(0))
mll = mlls.ExactMarginalLogLikelihood(osgpr_model.likelihood, osgpr_model)
trainable_params = [
    dict(params=osgpr_model.likelihood.parameters(), lr=1e-1),
    dict(params=osgpr_model.covar_module.parameters(), lr=1e-1),
    dict(params=osgpr_model.variational_strategy.inducing_points, lr=1e-2),
    dict(params=osgpr_model.variational_strategy._variational_distribution.parameters(), lr=1e-2)
]
optimizer = torch.optim.Adam(trainable_params)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400, 1e-4)



def upper_bound_single(model, x, y, P_dis, epsilon_square = 0.01):
    # calculate prior distribution:

    Q_mean, Q_var = model.predict_dis(x)
    Q_dis = Normal(Q_mean, torch.sqrt(Q_var))

    loss_exp_item = torch.div(torch.pow(torch.squeeze(y) - Q_dis.mean, 2),
                              2 * torch.sqrt(Q_dis.variance) + epsilon_square)
    loss_frac_item = torch.div(1, torch.sqrt(1 + 2 * Q_dis.variance / epsilon_square))
    loss_item = torch.mul(loss_frac_item, torch.exp(-loss_exp_item))
    kl_divergence = KL_div(Q_dis, P_dis)
    print(kl_divergence)
    empirical_risk = torch.mean(1 - loss_item)

    lembda = 1 / len(x)
    upper_bound = empirical_risk + kl_divergence / lembda

    return upper_bound


def upper_bound_Multi(model, x, y, P_dis, epsilon_square=0.01):
    Q_mean, Q_var = model.predict_dis(x)
    Q_dis = MVN(Q_mean, torch.diag(torch.sqrt(Q_var)))

    loss_exp_item = torch.div(torch.pow(torch.squeeze(y) - Q_dis.mean, 2),
                              2 * torch.sqrt(Q_dis.variance) + epsilon_square)
    loss_frac_item = torch.div(1, torch.sqrt(1 + 2 * Q_dis.variance / epsilon_square))
    loss_item = torch.mul(loss_frac_item, torch.exp(-loss_exp_item))
    kl_divergence = KL_div(Q_dis, P_dis)
    print(kl_divergence)
    empirical_risk = torch.mean(1 - loss_item)

    lembda = 1 / len(x)
    upper_bound = empirical_risk + kl_divergence / lembda

    return upper_bound


osgpr_model.train()
records = []
for i in range(400):
    optimizer.zero_grad()
    # loss = upper_bound(osgpr_model, init_x, P_dis=P_dis)
    train_dist = osgpr_model(init_x)
    loss = -elbo(train_dist, init_y.squeeze(-1)).sum()
    # loss = upper_bound_Multi(osgpr_model, init_x, init_y, P_dis_multi, epsilon_square=0.01)

    loss.backward()
    optimizer.step()
    lr_scheduler.step()

osgpr_model.eval()
osgpr_model = osgpr_model.get_fantasy_model(init_x, init_y, resample_ratio=0)






fig = plt.figure(figsize=(15, 4))
subplot_count = 1
# osgpr_model.set_lr(online_lr)
for t, (x, y) in enumerate(zip(X, Y)):
    if t % chunk_size == 0:
        inducing_x = osgpr_model.variational_strategy.inducing_points
        inducing_f = osgpr_model.variational_strategy.variational_distribution.loc
        train_x = torch.cat([init_x, X[:t + 1]])
        train_y = torch.cat([init_y, Y[:t + 1]])
        ax = fig.add_subplot(1, 3, subplot_count)
        if subplot_count == 1:
            ax.set_ylabel('$y$', rotation=0)
        subplot_count += 1

        ax = draw_plot(osgpr_model, train_x, train_y, X[t + 1:], Y[t + 1:], inducing_x, inducing_f, ax,
                       model_colors['sgpr'])

    # elbo = models.StreamingSGPRBound(osgpr_model)
    trainable_params = [
        dict(params=osgpr_model.likelihood.parameters(), lr=1e-2),
        dict(params=osgpr_model.covar_module.parameters(), lr=1e-2),
        dict(params=osgpr_model.variational_strategy.inducing_points, lr=1e-3),
    ]
    optimizer = torch.optim.Adam(trainable_params)
    # model.eval()
    # with torch.no_grad():
    #     mean, var = model.predict(x_grid)
    osgpr_model.eval()
    with torch.no_grad():
        mean, var = osgpr_model.predict_dis(x)
    P_dis = Normal(deepcopy(mean), deepcopy(torch.sqrt(var)))
    osgpr_model.train()

    for _ in range(2):
        optimizer.zero_grad()
        loss = upper_bound_single(osgpr_model, x, y, P_dis, epsilon_square = 0.01)
        loss.backward()
        optimizer.step()

    resample_ratio = 0.1 if t % 2 == 1 else 0
    osgpr_model = osgpr_model.get_fantasy_model(x.view(-1, 1), y.view(-1, 1), resample_ratio)


# fig.suptitle("O-SGPR, non-IID observations")
plt.subplots_adjust(top=0.80)
plt.savefig('./online_gp_pac_cos_iid.pdf')
# plt.savefig('./online_gp_pac_framework.pdf')
plt.show()




print("End online_gp_pac, time-series data")

print("End online_gp_pac")




