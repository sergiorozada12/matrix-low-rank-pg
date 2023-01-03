import torch
import gym

from src.utils import calculate_returns


def REINFORCE(env, agent, gamma=0.99, epochs=100, T=1000):
    totals, timesteps = [], []
    for epoch in range(epochs):
        states, actions, rewards = [], [], []

        s_t = env.reset()
        for t in range(T):
            a_t = agent.act(s_t)
            if isinstance(env.action_space, gym.spaces.Box):
                s_t_next, r_t, d_t, _ = env.step(a_t.numpy())
            else:
                s_t_next, r_t, d_t, _ = env.step(a_t.item())
            states.append(s_t)
            actions.append(a_t)
            rewards.append(r_t)

            if d_t:
                break

            s_t = s_t_next

        returns = calculate_returns(rewards, gamma)
        agent.learn(states, actions, returns)

        totals.append(sum(rewards))
        timesteps.append(t)

        print(f'{epoch}/{epochs}: {totals[-1]} - {t} \r', end='')

    return agent, totals, timesteps
