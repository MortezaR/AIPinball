import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

# PPO implementation
class RolloutMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        num_states = len(self.states)
        batch_start = np.arange(0,num_states, self.batch_size)
        indices = np.arange(num_states, dtype=np.int64)
        np.random.shuffle(indices)
        shuffled_batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), shuffled_batches


    def store_memory(self, state, actions, log_probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):

    def __init__(self,num_actions_dim,input_dims,alpha,
                 hidden_layer1_dim = 256,hidden_layer2_dim = 256, savefile_dir = 'tmp/ppo'):
        super().__init__()

        self.checkpoint_file = os.path.join(savefile_dir, 'actor_torch_ppo')

        self.actor = nn.Sequential(
            nn.Linear(input_dims, hidden_layer1_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer1_dim, hidden_layer2_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer2_dim, num_actions_dim),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state):
        probs = self.actor(state)
        distribution = Categorical(probs)

        return distribution

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self,input_dims,alpha,
                 hidden_layer1_dim = 256,hidden_layer2_dim = 256, savefile_dir = 'tmp/ppo'):
        super().__init__()

        self.checkpoint_file = os.path.join(savefile_dir, 'critic_torch_ppo')

        self.critic = nn.Sequential(
            nn.Linear(input_dims, hidden_layer1_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer1_dim, hidden_layer2_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer2_dim, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, num_actions_dim, input_dims, gamma =0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, steps_before_update=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda =  gae_lambda

        self.actor = ActorNetwork(num_actions_dim,input_dims,alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = RolloutMemory(batch_size)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("---saving models---")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("---loading models---")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        distribution = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()

        probs = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        dev = self.actor.device
        for epoch in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, values, reward_arr, dones_arr, batches = self.memory.generate_batches()

            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                advantage_at_t = 0
                for k in range(t, len(reward_arr) - 1):
                    advantage_at_t += discount * (reward_arr[k] + self.gamma*values[k+1]* (1-int(dones_arr[k])) - values[k])

                    discount *= self.gamma * self.gae_lambda
                advantage[t] = advantage_at_t
            advantage = torch.tensor(advantage).to(dev)

            values = torch.tensor(values).to(dev)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype= torch.float).to(dev)
                old_probs = torch.tensor(old_probs_arr[batch]).to(dev)
                actions = torch.tensor(action_arr[batch]).to(dev)

                distribution = self.actor(states)
                critic_value = torch.squeeze(self.critic(states))

                new_probs = distribution.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            self.memory.clear_memory()
