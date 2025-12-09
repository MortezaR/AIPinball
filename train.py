# import gymnasium as gym
# import pinball_env_vpx as pe
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from PPO_agent import PPOAgent
# from tqdm import tqdm
# 
# # ------------------ Make the env ------------------
# env = pe.PinballEnv(
#     server_url="http://127.0.0.1:5000",
#     frame_skip=2,
#     dt=1 / 60,
# )
# agent = PPOAgent(env.actionspace.n, observation_space.shape, gamma =0.99, alpha=0.0003, gae_lambda=0.95,
#                  policy_clip=0.2, batch_size=64, steps_before_update=2048, n_epochs=10)
# # ------------------ Training loop ------------------
# 
# n_updates = 100        # keep small at first so you see it work
# trajectories_per_update = 3
# 
# 
# for update_idx in tqdm(range(n_updates), desc="Training updates"):
#     obs_batch = []
#     action_batch = []
#     reward_batch = []
#     terminated_batch = []
#     next_obs_batch = []
# 
#     for traj_idx in range(trajectories_per_update):
#         try:
#             obs, info = env.reset()
#         except Exception as e:
#             print(f"[main] reset failed: {e}")
#             raise
# 
#         done = False
#         trajectory = []
#         rewards = []
# 
#         step_idx = 0
# 
#         while not done:
#             action_idx = agent.get_action(obs)
#             action_vec = agent._int_to_action_vec(action_idx)
# 
#             # This is what VPX should see in the bridge logs
#             # print(f"[traj {traj_idx}] step {step_idx}: action_idx={action_idx}, vec={action_vec}")
# 
#             try:
#                 next_obs, reward, terminated, truncated, info = env.step(action_vec)
#             except Exception as e:
#                 print(f"[main] env.step crashed: {e}")
#                 done = True
#                 break
# 
#             trajectory.append((obs, action_idx, reward, terminated, next_obs))
#             rewards.append(reward)
# 
#             obs = next_obs
#             done = terminated or truncated
#             step_idx += 1
# 
#         max_reward = max(rewards) if rewards else 0.0
# 
#         for (obs_t, action_t, _r, terminated_t, next_obs_t) in trajectory:
#             obs_batch.append(obs_t)
#             action_batch.append(action_t)
#             reward_batch.append(max_reward)
#             terminated_batch.append(terminated_t)
#             next_obs_batch.append(next_obs_t)
# 
#         agent.decay_epsilon()
# 
#     agent.update_batch(
#         obs_batch=obs_batch,
#         action_batch=action_batch,
#         reward_batch=reward_batch,
#         terminated_batch=terminated_batch,
#         next_obs_batch=next_obs_batch,
#     )
# 
#     if (update_idx + 1) % 5 == 0:
#         mean_reward = float(np.mean(reward_batch)) if reward_batch else 0.0
#         mean_td = agent.training_error[-1] if agent.training_error else 0.0
#         print(
#             f"[update {update_idx+1}] mean_reward={mean_reward:.3f}, "
#             f"mean_td_error={mean_td:.3f}, eps={agent.epsilon:.3f}"
#         )
# 
# # Optionally close env when done
# env.close()


import numpy as np
from pinball_env_vpx import PinballEnv  # adjust import to your file name
from tqdm.auto import trange  # or from tqdm import trange
from PPO_agent import Agent

def make_action_from_index(action_idx: int):
    mapping = {
        0: [0, 0],
        1: [1, 0],
        2: [0, 1],
        3: [1, 1],
    }
    return mapping[int(action_idx)]

def train_ppo(
    n_episodes: int = 20000,
    steps_per_update: int = 2048,
    max_steps_per_episode: int = 2000,
    save_every: int = 50,
):
    # ----- create env -----
    env = PinballEnv()
    obs_dim = env.observation_space.shape[0]  # should be len(obs_features)
    num_actions_dim = 4  # 4 discrete actions for the 2 flippers

    # ----- create agent -----
    agent = Agent(
        num_actions_dim=num_actions_dim,
        input_dims=obs_dim,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        steps_before_update=steps_per_update,
        n_epochs=10,
    )

    global_step = 0

    # tqdm progress bar over episodes
    episode_bar = trange(1, n_episodes + 1, desc="Training episodes")

    for episode in episode_bar:
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            # ----- select action -----
            action_idx, log_prob, value = agent.choose_action(obs)
            action = make_action_from_index(action_idx)

            # ----- step env -----
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ----- store transition -----
            agent.store_memory(
                state=obs,
                action=action_idx,
                probs=log_prob,
                vals=value,
                reward=reward,
                done=done,
            )

            episode_reward += reward
            obs = next_obs
            global_step += 1

            # ----- update PPO -----
            if global_step % steps_per_update == 0:
                agent.learn()

            if done:
                break

        # update tqdm bar text
        episode_bar.set_postfix(
            return_=f"{episode_reward:.2f}",
            steps=t + 1,
        )

        # occasionally save models
        if episode % save_every == 0:
            agent.save_models()

    # final update in case there is leftover data in memory
    if len(agent.memory.states) > 0:
        agent.learn()
        agent.memory.clear_memory()

    env.close()


if __name__ == "__main__":
    train_ppo(
        n_episodes=20000,
        steps_per_update=2048,
        max_steps_per_episode=100000,
        save_every=100,
    )
