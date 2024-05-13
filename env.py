import gymnasium as gym
import torch
from pyrcareworld.envs import RCareWorld


class MyEnvironment(gym.Env):
    def __init__(self):
        # Define your environment's properties here
        self.n_obs = 3
        self.n_actions = 2

    def reset(self):
        # Reset the environment to its initial state
        self.obs = torch.randn(self.n_obs)
        print("obs")
        print(self.obs)
        return self.obs

    def step(self, action):
        # Take a step in the environment based on the given action
        # return (self.obs.max() - self.obs[action] + torch.randn(1)).item()
        print(f"taking action: {action}")
        print("reward")
        print((self.obs[action] - self.obs.max()).item())
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
        self.n_obs = 3
        self.n_actions = 2
        self.robot = self.create_robot(
                id=639787, gripper_list=[639787], robot_name="franka", base_pos=[0, 0, 0]
            )
        self.target = self.create_object(id=2333, name="Cube", is_in_scene=True)
        self.elbow_marker = self.create_object(id=1001, name="Cube", is_in_scene=True)
        self.wrist_marker = self.create_object(id=1002, name="Cube", is_in_scene=True)
        self.comfortable_maker = self.create_object(id=1003, name="Cube", is_in_scene=True)

    def reset(self):
        self.instance_channel.set_action(
                "SetJointPositionDirectly",
                id=5050,
                joint_positions=[0],
            )
        for i in range(10):
            self._step()
        self.robot.setJointPositionsDirectly([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(5):
            self._step()
        position = self.target.getPosition()
        self.robot.BioIKMove(position)
        for i in range(20):
            self._step()
        
        elbow_position = self.elbow_marker.getPosition()
        wrist_position = self.wrist_marker.getPosition()

        wrist_pregrasp_position = wrist_position
        wrist_pregrasp_position[1] = wrist_pregrasp_position[1] - 0.05
        self.robot.BioIKMove(wrist_pregrasp_position)
        for i in range(20):
            self._step()

        self.robot.BioIKMove(wrist_position)
        for i in range(20):
            self._step()
        print("Human arm grasped.")

        
        self.obs = torch.tensor([wrist_position])
        print(self.obs)
        return self.obs

    def step(self, action):
        print(f"taking action: {action}")

        if action.item() == 0:
            self.robot.BioIKMove(targetPose=[0,0,-0.05], relative=True)
        else:
            self.robot.BioIKMove(targetPose=[0,0,0.05], relative=True)
        for i in range(20):
            self._step()
        wrist_position = self.wrist_marker.getPosition()
        comfort_marker_position = self.comfortable_maker.getPosition()
        comfort_level = -abs(wrist_position[2] - comfort_marker_position[2])
        print(f"comfort level:{comfort_level}")
        # make comfort_level a torch scalar
        comfort_level = torch.tensor(comfort_level)
        return comfort_level

    def close(self):
        self.env.close()
    
    