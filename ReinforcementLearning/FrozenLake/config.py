import os

class Config:
    def __init__(self):
        self.env_config = EnvConfig()
        self.env_config_human = EnvConfigHuman()
        self.dqn_config = DQNConfig()

class EnvConfig:
    def __init__(self):
        self.env_id = 'FrozenLake-v3'
        self.entry_point = 'gym.envs.toy_text:FrozenLakeEnv' # 실제 환경 클래스가 정의된 위치. Gym라이브러리에 내장된 FrozenLakeEnv 클래스 사용
        self.map_size = '4x4'    # '4x4' or '8x8'
        self.is_slippery = False
        self.render_mode = 'rgb_array' # 'rgb_array' or 'human

class EnvConfigHuman:
    def __init__(self):
        self.env_id = 'FrozenLake-v3_2'
        self.entry_point = 'gym.envs.toy_text:FrozenLakeEnv' # 실제 환경 클래스가 정의된 위치. Gym라이브러리에 내장된 FrozenLakeEnv 클래스 사용
        self.map_size = '4x4'    # '4x4' or '8x8'
        self.is_slippery = False
        self.render_mode = 'human' # 'rgb_array' or 'human

class DQNConfig:
    def __init__(self):
        self.num_episodes = 10000
        self.target_update = 10
        self.batch_size = 100

        # epsilon, 초반 20은 랜덤
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200

        self.gamma = 0.95

        self.limit_step = 20

        self.weight_path = os.getcwd() + "/weights"
        if not os.path.isdir(self.weight_path):
            os.makedirs(self.weight_path, exist_ok=True)

        self.negative_reward = -0.01
        self.positive_reward = 5
        self.same_pos_penalty = -0.01

        self.replay_memory_size = 1000

        # Optimizer 하이퍼파라미터
        self.lr = 2e-3 # 0.002
