import gymnasium as gym
import torch
from pyrcareworld.envs import RCareWorld


class MyEnvironment(gym.Env):
    def __init__(self):
        # Define your environment's properties here
        self.n_obs = 2
        self.n_actions = 2

    def reset(self):
        # Reset the environment to its initial state
        self.obs = torch.randn(self.n_obs)
        return self.obs

    def step(self, action):
        # Take a step in the environment based on the given action
        # return (self.obs.max() - self.obs[action] + torch.randn(1)).item()
        print(f"taking action: {action}")
        return (self.obs[action] - self.obs.max()).item()

    def close(self):
        # Clean up the environment's resources
        pass

class MyRCareWorldEnv(RCareWorld, gym.Env):
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        custom_channels: list = [],
        assets: list = [],
        **kwargs
    ):
        RCareWorld.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            custom_channels=custom_channels,
            assets=assets,
            **kwargs,
        )
        self.n_obs = 2
        self.n_actions = 2
        self.robot = self.create_robot(
                id=639787, gripper_list=[639787], robot_name="franka", base_pos=[0, 0, 0]
            )
        self.target = self.create_object(id=2333, name="Cube", is_in_scene=True)
        self.elbow_marker = self.create_object(id=1001, name="Cube", is_in_scene=True)
        self.wrist_marker = self.create_object(id=1002, name="Cube", is_in_scene=True)
        self.comfortable_maker = self.create_object(id=1003, name="Cube", is_in_scene=True)

    def reset(self):
        self.robot.setJointPositionsDirectly([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(5):
            self._step()
        position = self.target.getPosition()
        self.robot.BioIKMove(position, 3)
        for i in range(20):
            self._step()
        
        elbow_position = self.elbow_marker.getPosition()
        wrist_position = self.wrist_marker.getPosition()
        self.obs = torch.randn(self.n_obs)
        return self.obs

    def step(self, action):
        print(f"taking action: {action}")
        comfort_level = (self.obs[action] - self.obs.max()).item()
        print(f"comfort level:{comfort_level}")
        return (self.obs[action] - self.obs.max()).item()

    def close(self):
        self.env.close()
    
    