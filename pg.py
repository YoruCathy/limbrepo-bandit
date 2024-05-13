import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from env import MyEnvironment
import matplotlib.pyplot as plt

# hyperparameters below, can be tuned

T = 2000
lr = 3e-3


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def act(self, x):
        logits = self.model(x)
        action = torch.multinomial(F.softmax(logits, dim=-1), 1)
        return action, logits[action]


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    env = MyEnvironment()
    actor = Actor(env.n_obs, env.n_actions)
    critic = Critic(env.n_obs)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    reward_list = []

    for t in range(T):

        observation = env.reset()
        action, log_prob = actor.act(torch.tensor(observation).float())
        reward = env.step(action.long())

        print(f"round {t} \t reward {reward}")
        reward_list.append(reward)

        value = critic.forward(observation)
        advantage = reward - value
        pol_loss = -advantage.detach() * log_prob
        value_loss = (critic.forward(observation) - reward) ** 2

        loss = pol_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()

    # plot reward
    ma_reward = []
    for i in range(len(reward_list) - 20):
        ma_reward.append(np.mean(reward_list[i : i + 20]))
    plt.plot(ma_reward)
    plt.xlabel("round")
    plt.ylabel("reward")
    plt.show()
