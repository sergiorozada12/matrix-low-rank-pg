import pickle
import numpy as np


def load_data(path, k=1):
    with open(path,'rb') as f:
        data = pickle.load(f)

        median = np.median(data, axis=0)
        lower = np.percentile(data, 25, axis=0)
        upper = np.percentile(data, 75, axis=0)

        return median, lower, upper

if __name__ == "__main__":
    pend_nn, pend_nn_low, pend_nn_up = load_data('results/pend_nn.pkl')
    pend_lr, pend_lr_low, pend_lr_up = load_data('results/pend_lr.pkl')
    acro_nn, acro_nn_low, acro_nn_up = load_data('results/acro_nn.pkl')
    acro_lr, acro_lr_low, acro_lr_up = load_data('results/acro_lr.pkl')
    rock_nn, rock_nn_low, rock_nn_up = load_data('results/rock_nn.pkl', k=20)
    rock_lr, rock_lr_low, rock_lr_up = load_data('results/rock_lr.pkl', k=20)

    print("Pendulum LR: ", pend_lr[-1])
    print("Pendulum NN: ", pend_nn[-1])
    print("Acrobot LR: ", acro_lr[-1])
    print("Acrobot NN: ", acro_nn[-1])
    print("Rocket LR: ", rock_lr[-1])
    print("Rocket NN: ", rock_nn[-1])
