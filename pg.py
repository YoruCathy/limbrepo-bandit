import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from env import MyRCareWorldEnv
from env import MyEnvironment
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

# fix random seed
torch.manual_seed(123)
np.random.seed(123)


# hyperparameters below, can be tuned

T = 40
trials = 1
lr = 1e-2


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
    env = MyRCareWorldEnv()

    reward_list_list = []


    for trial in range(trials):
        actor = Actor(env.n_obs, env.n_actions)
        critic = Critic(env.n_obs)
        optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)



        print(f"===== trial {trial}")
        reward_list = []
        for t in tqdm(range(T)):
            # initialize environment


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

        # env.close()


        # ma_reward = []
        # for i in range(len(reward_list) - 20):
        #     ma_reward.append(np.mean(reward_list[i : i + 20]))
        # reward_list = ma_reward

        # plot reward
        if trial == 0:
            ma_reward = []
            for i in range(len(reward_list) - 20):
                ma_reward.append(np.mean(reward_list[i : i + 20]))
            plt.plot(ma_reward)
            plt.xlabel("round")
            plt.ylabel("reward")
            plt.show()

            os.makedirs("./models", exist_ok=True)
            mytime = time.time()
            torch.save(actor.state_dict(), f"./models/policy_{mytime}.pt")
        
        reward_list_list.append(reward_list)
    
    # plot reward with std
    reward_list_list = np.array(reward_list_list)
    mean_reward = np.mean(reward_list_list, axis=0)
    std_reward = np.std(reward_list_list, axis=0)
    plt.plot(mean_reward)
    plt.fill_between(range(len(mean_reward)), mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
    plt.xlabel("round")
    plt.ylabel("reward")
    plt.show()


    # inference / test
    inference_rounds = 5
    # mytime = time.time()
    actor.load_state_dict(torch.load(f"./models/policy_{mytime}.pt"))
    rewards = []
    for _ in range(inference_rounds):
        observation = env.reset()
        action, log_prob = actor.act(observation.float())
        reward = env.step(action.long())
        rewards.append(reward)
    print(f"mean reward : {np.mean(rewards)}")
    env.close()
