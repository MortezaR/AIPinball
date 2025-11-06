import gymnasium as gym
import pinball_env_vpx as pe

env = pe.PinballEnv(server_url="http://127.0.0.1:5000", action_schema="binary_flippers", frame_skip=2, dt=1/60)
obs, info = env.reset()
for _ in range(200):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    if term or trunc:
        break
env.close()
