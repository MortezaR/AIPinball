import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from PPO_agent import Agent


env = gym.make("CartPole-v1")

agent = Agent(
    num_actions_dim=env.action_space.n,
    input_dims=env.observation_space.shape[0],
    gamma=0.99,
    alpha=3e-4,
    gae_lambda=0.95,
    policy_clip=0.2,
    batch_size=64,
    steps_before_update=2048,
    n_epochs=10,
)

N = 20
n_episodes = 2000
global_step = 0
scores = []
avg_score = 0
learn_step = 0

for episode in range(n_episodes):
    # obs = env.reset()
    # gym<=0.25: obs = env.reset()
    obs, _ = env.reset()
    done = False
    score = 0

    while not done:
        action, log_prob, value = agent.choose_action(obs)
        # next_obs, reward, done, info = env.step(action)
        next_obs, reward, terminated, truncated, info = env.step(action); done = terminated or truncated
        global_step +=1
        score += reward
        agent.store_memory(obs, action, log_prob, value, reward, done)
        score += reward

        if global_step % N == 0:
            agent.learn()
            learn_step += 1
        obs = next_obs

    scores.append(score)
    if (episode + 1) % 10 == 0:
        avg = np.mean(scores[-10:])
        print(f"Episode {episode+1}, last-10 avg score: {avg:.1f}")

env.close()
