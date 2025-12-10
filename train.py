import numpy as np
from pinball_env_vpx import PinballEnv  # adjust import to your file name
from tqdm.auto import trange
from PPO_agent import Agent
from torch.utils.tensorboard import SummaryWriter  # <-- NEW


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
    load_from: str | None = None,          # optional checkpoint to load
    log_dir: str = "tmp/pinball_ppo",     # <-- NEW: TensorBoard log directory
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

    # ----- optionally LOAD a pretrained agent -----
    if load_from is not None:
        print(f"Loading PPO agent weights")
        agent.load_models()

    # ----- TensorBoard writer -----
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    running_return = None  # for smoothed return logging

    # tqdm progress bar over episodes
    episode_bar = trange(1, n_episodes + 1, desc="Training episodes")

    try:
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

            steps_this_episode = t + 1  # t is last step index

            # ----- TensorBoard logging -----
            writer.add_scalar("Episode/Return", episode_reward, episode)
            writer.add_scalar("Episode/Length", steps_this_episode, episode)
            writer.add_scalar("Training/GlobalStep", global_step, episode)

            # smoothed return (EMA)
            if running_return is None:
                running_return = episode_reward
            else:
                running_return = 0.9 * running_return + 0.1 * episode_reward
            writer.add_scalar("Episode/Return_Smoothed", running_return, episode)

            # update tqdm bar text
            episode_bar.set_postfix(
                return_=f"{episode_reward:.2f}",
                steps=steps_this_episode,
            )

            # occasionally save models
            if episode % save_every == 0:
                agent.save_models()

        # final update in case there is leftover data in memory
        if len(agent.memory.states) > 0:
            agent.learn()
            agent.memory.clear_memory()

    finally:
        # make sure writer is closed even if something crashes
        writer.close()
        env.close()


if __name__ == "__main__":
    train_ppo(
        n_episodes=20000,
        steps_per_update=2048,
        max_steps_per_episode=100000,
        save_every=20,
        load_from=True,
        log_dir="tmp/pinball_ppo",
    )
