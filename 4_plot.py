import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

ALPHA = .9
ALPHA_BACK = .2


def load_data(path, k=1):
    with open(path,'rb') as f:
        data = pickle.load(f)

        median = np.median(data, axis=0)
        lower = np.percentile(data, 25, axis=0)
        upper = np.percentile(data, 75, axis=0)

        upper = np.max(np.lib.stride_tricks.sliding_window_view(upper, k), axis=1)
        lower = np.min(np.lib.stride_tricks.sliding_window_view(lower, k), axis=1)

        pad_width = ((k)//2, (k-1)//2)
        lower = np.pad(lower, pad_width, mode='edge')
        upper = np.pad(upper, pad_width, mode='edge')

        return median, lower, upper

if __name__ == "__main__":
    pend_nn, pend_nn_low, pend_nn_up = load_data('results/pend_nn.pkl')
    pend_lr, pend_lr_low, pend_lr_up = load_data('results/pend_lr.pkl')
    acro_nn, acro_nn_low, acro_nn_up = load_data('results/acro_nn.pkl')
    acro_lr, acro_lr_low, acro_lr_up = load_data('results/acro_lr.pkl')
    rock_nn, rock_nn_low, rock_nn_up = load_data('results/rock_nn.pkl', k=20)
    rock_lr, rock_lr_low, rock_lr_up = load_data('results/rock_lr.pkl', k=20)

    pend_time = np.arange(pend_nn.size)
    acro_time = np.arange(acro_nn.size)
    rock_time = np.arange(rock_nn.size)

    with plt.style.context(['science'], ['ieee']):
        fig, ax = plt.subplots(3, figsize=[8, 12])

        plt.rc('legend', fontsize=16)    # legend fontsize

        ax[0].plot(pend_time, pend_lr, color='g', label='LRPG - 160 params.', alpha=ALPHA)
        ax[0].fill_between(pend_time, pend_lr_low, pend_lr_up, color='g', alpha=ALPHA_BACK)
        ax[0].plot(pend_time, pend_nn, color='r', label='RVFB - 4098 params.', alpha=ALPHA)
        ax[0].fill_between(pend_time, pend_nn_low, pend_nn_up, color='r', alpha=ALPHA_BACK)
        ax[0].set_ylabel('(a) Return', fontsize=18)
        ax[0].set_xlabel('Episodes', fontsize=18)
        ax[0].set_xlim(0, 1000)
        ax[0].legend(loc='lower right')
        ax[0].set_xticks([0, 250, 500, 750, 1000])
        ax[0].set_yticks([25, 50, 75, 100])
        ax[0].tick_params(axis='both', which='major', labelsize=14)
        ax[0].grid()

        ax[1].plot(acro_time, acro_lr, color='g', label='LRPG - 16 params.', alpha=ALPHA)
        ax[1].fill_between(acro_time, acro_lr_low, acro_lr_up, color='g', alpha=ALPHA_BACK)
        ax[1].plot(acro_time, acro_nn, color='r', label='RVFB - 386 params.', alpha=ALPHA)
        ax[1].fill_between(acro_time, acro_nn_low, acro_nn_up, color='r', alpha=ALPHA_BACK)
        ax[1].set_ylabel('(b) Return', fontsize=18)
        ax[1].set_xlabel('Episodes', fontsize=18)
        ax[1].set_xlim(0, 1000)
        ax[1].set_ylim(0, 100)
        ax[1].legend(loc='lower right')
        ax[1].set_xticks([0, 250, 500, 750, 1000], fontsize=14)
        ax[1].set_yticks([25, 50, 75, 100], fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=14)
        ax[1].grid()

        ax[2].plot(rock_time, rock_lr, color='g', label='LRPG - 80', alpha=ALPHA)
        ax[2].fill_between(rock_time, rock_lr_low, rock_lr_up, color='g', alpha=ALPHA_BACK)
        ax[2].plot(rock_time, rock_nn, color='r', label='RVFB - 1282 params.', alpha=ALPHA)
        ax[2].fill_between(rock_time, rock_nn_low, rock_nn_up, color='r', alpha=ALPHA_BACK)
        ax[2].set_ylabel('(c) Return', fontsize=18)
        ax[2].set_xlabel('Episodes', fontsize=18)
        ax[2].set_xlim(0, 30000)
        ax[2].set_ylim(0.7, 1.25)
        ax[2].legend(loc='lower right')
        ax[2].set_xticks([0, 10000, 20000, 30000], fontsize=14)
        ax[2].set_yticks([0.8, 1.0, 1.2], fontsize=14)
        ax[2].tick_params(axis='both', which='major', labelsize=14)
        ax[2].grid()

        plt.tight_layout()
        fig.savefig('figures/fig1.png', dpi=300)
