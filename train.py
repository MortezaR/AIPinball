import gymnasium as gym
import pinball_env_vpx as pe
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# ------------------ Make the env ------------------
env = pe.PinballEnv(
    server_url="http://127.0.0.1:5000",
    frame_skip=2,
    dt=1 / 60,
)


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
        env: gym.Env,
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

        # --- Observation / action dimensions ---
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 1, f"Expected 1D obs, got {obs_shape}"
        input_dim = obs_shape[0]

        # discrete actions -> MultiBinary(4)
        self.action_table = np.array(
            [
                [0, 0, 0, 0],  # 0: do nothing
                [1, 0, 0, 0],  # 1: left flipper
                [0, 1, 0, 0],  # 2: right flipper
                [0, 0, 0, 1],  # 3: plunger
            ],
            dtype=np.int32,
        )
        self.n_actions = self.action_table.shape[0]
        output_dim = self.n_actions

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_network = QNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        print(f"[agent] obs_dim = {input_dim}, n_actions = {self.n_actions}, device = {self.device}")

    # ---------- utilities ----------
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def _encode_obs(self, obs):
        """Pinball obs is already a vector; just convert to tensor."""
        x = np.asarray(obs, dtype=np.float32)
        return torch.tensor(x, device=self.device)

    def _int_to_action_vec(self, a_int: int) -> np.ndarray:
        return self.action_table[a_int]

    # ---------- acting ----------
    def get_action(self, obs):
        """
        Epsilon-greedy:
          - with prob epsilon: random action
          - else: sample from softmax over Q(s,·)
        """
        if np.random.rand() < self.epsilon:
            a = int(np.random.randint(self.n_actions))
            # print(f"[agent] random action {a}")
            return a

        with torch.no_grad():
            x = self._encode_obs(obs)
            q_values = self.q_network(x)          # (n_actions,)
            probs = torch.softmax(q_values, dim=-1)
            a = torch.multinomial(probs, num_samples=1).item()
            # print(f"[agent] policy action {a}, q={q_values.cpu().numpy()}")
            return int(a)

    # ---------- learning ----------
    def update_batch(
        self,
        obs_batch,        # list[np.ndarray]
        action_batch,     # list[int]
        reward_batch,     # list[float]
        terminated_batch, # list[bool]
        next_obs_batch,   # list[np.ndarray]
    ):
        """Batched Q-learning update over a whole batch of transitions."""
        if not obs_batch:
            return

        s = torch.stack([self._encode_obs(o) for o in obs_batch], dim=0)        # (B, obs_dim)
        s_next = torch.stack([self._encode_obs(o) for o in next_obs_batch], 0)  # (B, obs_dim)

        actions = torch.tensor(action_batch, dtype=torch.long, device=self.device)         # (B,)
        rewards = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)      # (B,)
        terminated = torch.tensor(terminated_batch, dtype=torch.float32, device=self.device)  # (B,)

        q_values = self.q_network(s)                         # (B, n_actions)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            q_next = self.q_network(s_next)                  # (B, n_actions)
            max_q_next, _ = q_next.max(dim=1)                # (B,)
            target = rewards + (1.0 - terminated) * self.discount_factor * max_q_next

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = (target - q_sa).detach().cpu().numpy()
        self.training_error.append(float(np.mean(td_errors)))

    # ---------- checkpointing ----------
    def save(self, path: str):
        checkpoint = {
            "model_state": self.q_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)
        print(f"[agent] checkpoint saved to {path}")

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.q_network.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        print(f"[agent] checkpoint loaded from {path}")


# ------------------ Training loop ------------------

n_updates = 100        # keep small at first so you see it work
trajectories_per_update = 3

agent = PinballAgent(
    env=env,
    learning_rate=1e-3,
    initial_epsilon=1.0,   # start fully random so we see movement
    epsilon_decay=0.99,
    final_epsilon=0.1,
)

for update_idx in tqdm(range(n_updates), desc="Training updates"):
    obs_batch = []
    action_batch = []
    reward_batch = []
    terminated_batch = []
    next_obs_batch = []

    for traj_idx in range(trajectories_per_update):
        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"[main] reset failed: {e}")
            raise

        done = False
        trajectory = []
        rewards = []

        step_idx = 0

        while not done:
            action_idx = agent.get_action(obs)
            action_vec = agent._int_to_action_vec(action_idx)

            # This is what VPX should see in the bridge logs
            # print(f"[traj {traj_idx}] step {step_idx}: action_idx={action_idx}, vec={action_vec}")

            try:
                next_obs, reward, terminated, truncated, info = env.step(action_vec)
            except Exception as e:
                print(f"[main] env.step crashed: {e}")
                done = True
                break

            trajectory.append((obs, action_idx, reward, terminated, next_obs))
            rewards.append(reward)

            obs = next_obs
            done = terminated or truncated
            step_idx += 1

        max_reward = max(rewards) if rewards else 0.0

        for (obs_t, action_t, _r, terminated_t, next_obs_t) in trajectory:
            obs_batch.append(obs_t)
            action_batch.append(action_t)
            reward_batch.append(max_reward)
            terminated_batch.append(terminated_t)
            next_obs_batch.append(next_obs_t)

        agent.decay_epsilon()

    agent.update_batch(
        obs_batch=obs_batch,
        action_batch=action_batch,
        reward_batch=reward_batch,
        terminated_batch=terminated_batch,
        next_obs_batch=next_obs_batch,
    )

    if (update_idx + 1) % 5 == 0:
        mean_reward = float(np.mean(reward_batch)) if reward_batch else 0.0
        mean_td = agent.training_error[-1] if agent.training_error else 0.0
        print(
            f"[update {update_idx+1}] mean_reward={mean_reward:.3f}, "
            f"mean_td_error={mean_td:.3f}, eps={agent.epsilon:.3f}"
        )

# Optionally close env when done
env.close()
