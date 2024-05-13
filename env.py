import gym
import torch


class MyEnvironment(gym.Env):
    def __init__(self):
        # Define your environment's properties here
        self.n_obs = 5
        self.n_actions = self.n_obs

    def reset(self):
        # Reset the environment to its initial state
        self.obs = torch.randn(self.n_obs)
        return self.obs

    def step(self, action):
        # Take a step in the environment based on the given action
        # return (self.obs.max() - self.obs[action] + torch.randn(1)).item()
        return (self.obs[action] - self.obs.max()).item()

    def close(self):
        # Clean up the environment's resources
        pass
