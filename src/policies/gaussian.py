import numpy as np
import torch

from src.models.matrix import LR


class GaussianPolicyLR:
    def __init__(
        self,
        env,
        discretizer_actor,
        discretizer_critic,
        k=1,
        lr_actor=1e-2,
        lr_critic=1e-2,
        scale=1.0,
        bool_output=False
    ):
        self.discretizer_actor = discretizer_actor
        self.discretizer_critic = discretizer_critic

        self.bool_output = bool_output

        self.N_actor = discretizer_actor.N
        self.M_actor = discretizer_actor.M

        self.N_critic = discretizer_critic.N
        self.M_critic = discretizer_critic.M

        self.S = env.reset().shape[0]

        self.mu = LR(self.N_actor, self.M_actor, k, scale).double()
        self.value = LR(self.N_critic, self.M_critic, k, scale).double()

        self.log_sigma = torch.ones(1, dtype=torch.double, requires_grad=True)

        self.opt_actor = torch.optim.Adam(list(self.mu.parameters()) + [self.log_sigma], lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.value.parameters(), lr=lr_critic)

    def v(self, s_t):
        s_t = s_t.reshape(-1, self.S)
        rows, cols = self.discretizer_critic.get_index(s_t)
        return self.value(rows, cols)

    def pi(self, s_t):
        s_t = s_t.reshape(-1, self.S)
        rows, cols = self.discretizer_actor.get_index(s_t)

        mu = self.mu(rows, cols)

        if self.bool_output:
            mu = torch.nn.Sigmoid()(mu)

        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)

        pi = torch.distributions.Normal(mu, sigma)
        return pi

    def act(self, s_t):
        a_t = self.pi(s_t).sample()
        return torch.clamp(a_t, 0.0, 1.0) if self.bool_output else a_t

    def learn(self, states, actions, returns):
        returns = torch.tensor(returns)
        states = np.array(states)

        values = self.v(states)
        with torch.no_grad():
            advantages = returns - values

        # Actor
        log_prob = self.pi(states).log_prob(actions)
        loss_action = torch.mean(-log_prob*advantages)
        self.opt_actor.zero_grad()
        loss_action.backward()
        self.opt_actor.step()

        # Critic
        loss_fn = torch.nn.MSELoss()
        loss_value = loss_fn(values.double(), returns.double())
        self.opt_critic.zero_grad()
        loss_value.backward()
        self.opt_critic.step()


class GaussianPolicyNN:
    def __init__(self, env, mu, v, lr_actor=1e-2, lr_critic=1e-2, bool_output=False):
        self.bool_output = bool_output

        self.mu = mu
        self.value = v

        self.log_sigma = torch.ones(1, dtype=torch.double, requires_grad=True)
        self.opt_actor = torch.optim.Adam(list(self.mu.parameters()) + [self.log_sigma], lr=lr_actor) 
        self.opt_critic = torch.optim.Adam(self.value.parameters(), lr=lr_critic)

    def pi(self, s_t):
        s_t = torch.as_tensor(s_t).double()
        mu = self.mu(s_t)

        if self.bool_output:
            mu = torch.nn.Sigmoid()(mu)

        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        return pi

    def v(self, s_t):
        s_t_tensor = torch.as_tensor(s_t).double()
        return self.value(s_t_tensor)

    def act(self, s_t):
        a_t = self.pi(s_t).sample()
        return torch.clamp(a_t, 0.0, 1.0) if self.bool_output else a_t

    def learn(self, states, actions, returns):
        returns = torch.tensor(returns)
        states = torch.tensor(states)

        values = self.v(states)
        with torch.no_grad():
            advantages = returns - values

        # Actor
        log_prob = self.pi(states).log_prob(actions)
        loss_action = torch.mean(-log_prob*advantages)
        self.opt_actor.zero_grad()
        loss_action.backward()
        self.opt_actor.step()

        # Critic
        loss_fn = torch.nn.MSELoss()
        loss_value = loss_fn(values.double(), returns.reshape(-1, 1).double())
        self.opt_critic.zero_grad()
        loss_value.backward()
        self.opt_critic.step()


def get_nn_policy(env, mu, v, lr_actor, lr_critic, bool_output=False):
    return GaussianPolicyNN(
            env,
            mu,
            v,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            bool_output=bool_output
        )


def get_lr_policy(env, discretizer_actor, discretizer_critic, k, lr_actor, lr_critic, bool_output=False):
    return GaussianPolicyLR(
        env,
        discretizer_actor,
        discretizer_critic,
        k=k,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        bool_output=bool_output
    )
