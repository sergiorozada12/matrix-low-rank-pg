import pickle
import torch

from src.policies.gaussian import get_lr_policy, get_nn_policy
from src.environments.pendulum import CustomPendulumEnv
from src.algorithms.reinforce import REINFORCE
from src.utils import Discretizer


def get_model():
    return torch.nn.Sequential(
            torch.nn.Linear(2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        ).double()

def get_discretizer(bucket):
    Discretizer(
        min_points=[-1, -5],
        max_points=[1, 5],
        buckets=[bucket, bucket],
        dimensions=[[0], [1]]
    )

if __name__ == "__main__":
    env = CustomPendulumEnv()

    reward_nn = []
    reward_lr = []
    for _ in range(100):
        # Neural Network
        mu = get_model()
        v = get_model()
        agent = get_nn_policy(env, mu, v, lr_actor=1e-4, lr_critic=1e-4)
        _, totals, _ = REINFORCE(env, agent, gamma=0.99, epochs=5000, T=100)
        reward_nn.append(totals)

        # Low-rank matrix
        discretizer_actor = get_discretizer(10)
        discretizer_critic = get_discretizer(16)
        agent = get_lr_policy(env, discretizer_actor, discretizer_critic, lr_actor=2e-3, lr_critic=1e-1)
        _, totals, _ = REINFORCE(env, agent, gamma=0.99, epochs=5000, T=100)
        reward_lr.append(totals)

    with open('results/pend_nn.pkl','wb') as f:
        pickle.dump(reward_nn, f)

    with open('results/pend_lr.pkl','wb') as f:
        pickle.dump(reward_lr, f)
