import gymnasium as gym
import pinball_env_vpx as pe
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
env = pe.PinballEnv(server_url="http://127.0.0.1:5000", action_schema="binary_flippers", frame_skip=2, dt=1/60)

# obs, info = env.reset()
# for _ in range(200):
#     a = env.action_space.sample()
#     obs, r, term, trunc, info = env.step(a)
#     if term or trunc:
#         break
# env.close()


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PinballAgent:
    def __init__(
        self,
        env: env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        hidden_dim: int = 64,
        device: str | None = None,
    ):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

        # NN setup
        input_dim = 3  # player_sum, dealer_card, usable_ace
        output_dim = env.action_space.n

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_network = QNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    # Encode obs
    def _encode_obs(self, obs):
        x = np.array([obs[0], obs[1], float(obs[2])], dtype=np.float32)
        return torch.tensor(x, device=self.device)

    def get_action(self, obs):

        with torch.no_grad():
            x = self._encode_obs(obs)
            q_values = self.q_network(x)
            #sample from dis instead of argmax
            probs = torch.softmax(q_values, dim=-1)  # Convert Q-values to probabilities
            action = torch.multinomial(probs, num_samples=1).item()  # Sample based on probs
            return int(action)

    def update_batch(
            self,
            obs_batch,  # list[tuple[int, int, bool]]
            action_batch,  # list[int]
            reward_batch,  # list[float]
            terminated_batch,  # list[bool]
            next_obs_batch,  # list[tuple[int, int, bool]]
        ):
            """Batched Q-learning update over a whole batch of transitions."""

            # Convert to tensors
            # shape: (B, 3)
            s = torch.stack([self._encode_obs(o) for o in obs_batch], dim=0)  # (B, 3)
            s_next = torch.stack([self._encode_obs(o) for o in next_obs_batch], dim=0)

            actions = torch.tensor(action_batch, dtype=torch.long, device=self.device)  # (B,)
            rewards = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)  # (B,)
            terminated = torch.tensor(terminated_batch, dtype=torch.float32, device=self.device)  # (B,)

            # Q(s, a) for all transitions
            q_values = self.q_network(s)  # (B, n_actions)
            # gather Q(s,a) for each chosen action
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

            # Max_a' Q(s', a') for all next states
            with torch.no_grad():
                q_next = self.q_network(s_next)  # (B, n_actions)
                max_q_next, _ = q_next.max(dim=1)  # (B,)

                # target = r + gamma * (1 - done) * max_a' Q(s', a')
                target = rewards + (1.0 - terminated) * self.discount_factor * max_q_next

            loss = self.loss_fn(q_sa, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track mean TD error (mostly for debugging)
            td_errors = (target - q_sa).detach().cpu().numpy()
            self.training_error.append(float(np.mean(td_errors)))


from tqdm import tqdm

n_updates = 1000              # how many parameter updates you want
trajectories_per_update = 10  # N trajectories before each update

for update_idx in tqdm(range(n_updates)):
    # big batch containers
    obs_batch = []
    action_batch = []
    reward_batch = []
    terminated_batch = []
    next_obs_batch = []

    # ---- collect N trajectories ----
    for _ in range(trajectories_per_update):
        obs, info = env.reset()
        done = False

        trajectory = []   # (obs, action, reward, terminated, next_obs)
        rewards = []

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append((obs, action, reward, terminated, next_obs))
            rewards.append(reward)

            obs = next_obs
            done = terminated or truncated

        # If no rewards (weird edge case), set to 0.0
        max_reward = max(rewards) if rewards else 0.0

        # Assign the SAME reward to all steps in this trajectory
        for (obs_t, action_t, _r, terminated_t, next_obs_t) in trajectory:
            obs_batch.append(obs_t)
            action_batch.append(action_t)
            reward_batch.append(max_reward)
            terminated_batch.append(terminated_t)
            next_obs_batch.append(next_obs_t)

        # Optional: decay epsilon once per trajectory
        agent.decay_epsilon()

    # ---- do one batched update over all collected transitions ----
    agent.update_batch(
        obs_batch=obs_batch,
        action_batch=action_batch,
        reward_batch=reward_batch,
        terminated_batch=terminated_batch,
        next_obs_batch=next_obs_batch,
    )

    #save the agent in the loop
    #save previous states of the agent and log/graph the loss/reward (print to terminal for now)
    #env reset get that working

    # ---------- Checkpointing ----------
    def save(self, path: str):
        checkpoint = {
            "model_state": self.q_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)

        self.q_network.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

        print(f"Checkpoint loaded from {path}")