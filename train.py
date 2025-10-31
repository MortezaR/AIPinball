# train.py
from gymnasium.envs.registration import register
import gymnasium as gym
import pinball_env

# register environment
register(
    id="PinballEnv-v0",
    entry_point="pinball_env:PinballEnv"
)

# create environment
env = gym.make("PinballEnv-v0")

# simple training loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
