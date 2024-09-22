import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.CELU(),
            nn.Linear(128, 128),
            nn.CELU(),
            nn.Linear(128, 128),
            nn.CELU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.CELU(),
            nn.Linear(128, 128),
            nn.CELU(),
            nn.Linear(128, 128),
            nn.CELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

# Hyperparameters
env_name = "CartPole-v1"
gamma = 0.99
epsilon = 0.2
learning_rate = 0.001
num_episodes = 1000
batch_size = 5
update_epochs = 4

# Environment setup
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize network, optimizer, and loss
model = ActorCritic(obs_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to compute returns
def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    Gt = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            Gt = 0
        Gt = reward + gamma * Gt
        returns.insert(0, Gt)
    return returns

# Main PPO Training Loop
for episode in range(num_episodes):
    obs = env.reset()[0]
    obs = torch.tensor(obs, dtype=torch.float32)
    
    log_probs = []
    values = []
    rewards = []
    dones = []
    states = []
    actions = []

    # Collect trajectory
    for _ in range(batch_size):
        action_probs, value = model(obs)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        next_obs, reward, done, _, _ = env.step(action.item())
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        
        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        states.append(obs)
        actions.append(action)
        
        obs = next_obs
        if done:
            obs = env.reset()[0]
            obs = torch.tensor(obs, dtype=torch.float32)
    
    # Convert lists to tensors
    log_probs = torch.stack(log_probs)
    values = torch.stack(values).squeeze()
    returns = compute_returns(rewards, dones, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - values.detach()

    # PPO update
    for _ in range(update_epochs):
        # Calculate new log probabilities and state values
        action_probs, state_values = model(torch.stack(states))
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(torch.stack(actions))
        
        # Ratio for PPO clipping
        ratios = torch.exp(new_log_probs - log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

        # Actor-Critic loss
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.functional.mse_loss(state_values.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print progress
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item()}")

env.close()
