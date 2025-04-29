import gym
from gym.envs.registration import register
import torch
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, env_config):
        self._config = env_config

        self._env_id = self._config.env_id
        self._entry_point = self._config.entry_point
        self._map_size = self._config.map_size
        self._is_slippery = self._config.is_slippery
        self._render_mode = self._config.render_mode

        self._register_environment(self._env_id, self._entry_point, self._map_size, self._is_slippery)
        self._env = self._create_env(self._env_id, self._render_mode)

    def _register_environment(self, env_id, entry_point, map_size, is_slippery):
        register(
            id=env_id,
            entry_point=entry_point,  
            kwargs={'map_name' : map_size, 'is_slippery': is_slippery}
        )

    def _create_env(self, env_id, render_mode):
        return gym.make(id=env_id, render_mode=render_mode)
    
    def step(self, action):
        return self._env.step(action)
    
    def reset(self):
        self._env.reset()

    def render(self):
        return self._env.render()

    def get_env_info(self):
        if self._render_mode != 'rgb_array':
            return None, None, None
        
        self.reset()
        screen = self._env.render().transpose((2,0,1))
        _, h, w = screen.shape
        n_actions = self._env.action_space.n

        return h, w, n_actions

    def get_screen(self):
        screen = self._env.render()   # 256, 256, 3   # (H,W,C) -> Torch Order (C,H,W)
        # screen_transposed = screen.transpose((2,0,1))   # 256, 256, 3   # (H,W,C) -> Torch Order (C,H,W)
        # screen_transposed = torch.from_numpy(screen_transposed).float()/255.0
        # screen_transposed = torch.from_numpy(screen_transposed).float()
        # return screen, screen_transposed.unsqueeze(0)
        return screen
    
    def start(self, mode=False):
        self._env.reset()
        self._env.render()
        if mode:
            while True:
                action = input("Enter action: ")
                if action not in ['0','1','2','3']:
                    continue
                action = int(action)
                state, reward, done, info, _prob_dict = self._env.step(action)
                self._env.render()
                print("State :", state, "Action: ", action, "Reward: ", reward, "info: ", info)
                print()
                if done:
                    print("Finished with reward", reward)
                    break
        else:
            print('no')
            self._env.render()

if __name__ == "__main__":
    from config import Config
    config = Config()
    env = Environment(config.env_config_human)
    env.reset()
    while True:
        env.render()
        action = input("{0: 'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'} >> ")
        # 3:right
        action = int(action)
        env.step(action)