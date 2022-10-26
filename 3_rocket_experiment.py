import pickle
import torch

from src.policies.gaussian import get_lr_policy, get_nn_policy
from src.environments.rocket import CustomGoddardEnv
from src.algorithms.reinforce import REINFORCE
from src.utils import Discretizer


def get_model():
    return torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1)
    ).double()

def get_discretizer():
    return Discretizer(
        min_points=[0.0, 1.00, 0.6],
        max_points=[0.12, 1.03, 1],
        buckets=[4, 4, 4],
        dimensions=[[0], [1, 2]]
    )

if __name__ == "__main__":
    env = CustomGoddardEnv()
    discretizer_actor = get_discretizer()
    discretizer_critic = get_discretizer()

    reward_nn = []
    reward_lr = []
    for _ in range(100):
        # Neural Network
        mu = get_model()
        v = get_model()
        agent = get_nn_policy(env, mu, v, lr_actor=1e-4, lr_critic=1e-4)
        _, totals, _ = REINFORCE(env, agent, gamma=0.99, epochs=15_000, T=1_000)
        reward_nn.append(totals)

        # Low-rank matrix
        agent = get_lr_policy(env, discretizer_actor, discretizer_critic, lr_actor=8e-3, lr_critic=1e-1)
        _, totals, _ = REINFORCE(env, agent, gamma=0.99, epochs=15_000, T=1_000)
        reward_lr.append(totals)

    with open('results/rock_nn.pkl','wb') as f:
        pickle.dump(reward_nn, f)

    with open('results/rock_lr.pkl','wb') as f:
        pickle.dump(reward_lr, f)
