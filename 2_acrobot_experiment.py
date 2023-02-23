import pickle
import torch

from src.policies.gaussian import get_lr_policy, get_nn_policy
from src.environments.acrobot import CustomAcrobotEnv
from src.algorithms.reinforce import REINFORCE
from src.utils import Discretizer


def get_model():
    return torch.nn.Sequential(
        torch.nn.Linear(4, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    ).double()

def get_discretizer():
    return Discretizer(
        min_points=[-1, -1, -1, -1],
        max_points=[1, 1, 1, 1],
        buckets=[2, 2, 2, 2],
        dimensions=[[0, 1], [2, 3]]
    )

if __name__ == "__main__":
    env = CustomAcrobotEnv()
    discretizer_actor = get_discretizer()
    discretizer_critic = get_discretizer()

    reward_nn = []
    reward_lr = []
    for _ in range(100):
        # Neural Network
        mu = get_model()
        v = get_model()
        agent = get_nn_policy(env, mu, v, lr_actor=1e-3, lr_critic=1e-3)
        _, totals, _ = REINFORCE(env, agent, gamma=0.9, epochs=1000, T=1000)
        reward_nn.append(totals)

        # Low-rank matrix
        agent = get_lr_policy(env, discretizer_actor, discretizer_critic, k=2, lr_actor=3e-2, lr_critic=1e-1)
        _, totals, _ = REINFORCE(env, agent, gamma=0.9, epochs=1000, T=1000)
        reward_lr.append(totals)

    with open('results/acro_nn.pkl','wb') as f:
        pickle.dump(reward_nn, f)

    with open('results/acro_lr.pkl','wb') as f:
        pickle.dump(reward_lr, f)
