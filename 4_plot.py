import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

ALPHA = .9


def load_data(path):
    with open(path,'rb') as f:
        return np.median(pickle.load(f), axis=0)

if __name__ == "__main__":
    pend_nn = load_data('results/pend_nn.pkl')
    pend_lr = load_data('results/pend_lr.pkl')
    acro_nn = load_data('results/acro_nn.pkl')
    acro_lr = load_data('results/acro_lr.pkl')
    rock_nn = load_data('results/rock_nn.pkl')
    rock_lr = load_data('results/rock_lr.pkl')

    pend_time = np.arange(pend_nn.size)
    acro_time = np.arange(acro_nn.size)
    rock_time = np.arange(rock_nn.size)

    fig, ax = plt.subplots(3, figsize=[8, 12])

    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('legend', fontsize=16)    # legend fontsize

    ax[0].plot(pend_time, pend_lr, color='g', label='LRPG - 160 params.', alpha=ALPHA)
    ax[0].plot(pend_time, pend_nn, color='r', label='RVFB - 4098 params.', alpha=ALPHA)
    ax[0].set_ylabel('(a) Return')
    ax[0].set_xlabel('Episodes')
    ax[0].set_xlim(0, 1000)
    ax[0].legend()

    ax[1].plot(acro_time, acro_lr, color='g', label='LRPG - 16 params.', alpha=ALPHA)
    ax[1].plot(acro_time, acro_nn, color='r', label='RVFB - 386 params.', alpha=ALPHA)
    ax[1].set_ylabel('(b) Return')
    ax[1].set_xlabel('Episodes')
    ax[1].set_xlim(0, 500)
    ax[1].legend()

    ax[2].plot(rock_time, rock_lr, color='g', label='LRPG - 80', alpha=ALPHA)
    ax[2].plot(rock_time, rock_nn, color='r', label='RVFB - 1282 params.', alpha=ALPHA)
    ax[2].set_ylabel('(c) Return')
    ax[2].set_xlabel('Episodes')
    ax[2].set_xlim(0, 30000)
    ax[2].legend()

    plt.tight_layout()
    fig.savefig('figures/fig2.png', dpi=300)
