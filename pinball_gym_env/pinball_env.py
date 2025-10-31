import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(MyEnv, self).__init__()

        # Define action and observation space
        # Example: agent can move left or right (discrete 2 actions)
        self.action_space = spaces.Discrete(2)

        # Example: observation is a single float between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Initialize state
        self.state = np.array([0.5], dtype=np.float32)
        self.render_mode = render_mode
        self.steps = 0

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state"""
        super().reset(seed=seed)
        self.state = np.array([0.5], dtype=np.float32)
        self.steps = 0
        info = {}
        return self.state, info

    def step(self, action):
        """Applies an action and returns observation, reward, done, info"""
        # Example: move state up/down depending on action
        if action == 0:
            self.state -= 0.05
        else:
            self.state += 0.05

        self.state = np.clip(self.state, 0, 1)
        self.steps += 1

        # Reward: closer to 1 is better
        reward = float(self.state[0])
        terminated = bool(self.state[0] >= 1.0)
        truncated = self.steps >= 100  # stop after 100 steps
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self):
        """Optional: visualize the environment"""
        if self.render_mode == "human":
            print(f"Current state: {self.state[0]:.2f}")

    def close(self):
        """Clean up resources"""
        pass
