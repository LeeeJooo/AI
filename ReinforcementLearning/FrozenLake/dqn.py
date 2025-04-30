from network import Network
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

Observation = namedtuple('Observation', 'cur_pos, cur_state, action, next_pos, next_state, reward, done')
class DQN:
    def __init__(self, config, env=None, animation_mode=True, policy_model_name=None, target_model_name=None):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._config = config
        
        self._env = env
        self.h, self.w, self.n_actions = self._env.get_env_info()
        self._map_size = int(self._config.env_config.map_size[0])

        self.policy_net = Network(self._map_size, self._map_size, self.n_actions).to(self._device)
        self.optimizer = optim.NAdam(params=self.policy_net.parameters(), lr=self._config.dqn_config.lr)               # lr = 2e-3 (default)
        
        self.target_net = Network(self._map_size, self._map_size, self.n_actions).to(self._device)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self._animation_mode = animation_mode
        if self._animation_mode:
            self._fig, self._ax = plt.subplots()
        self._fig_eval, self._ax_eval = plt.subplots(figsize=(self._map_size*1.8, self._map_size * 1.5))
        plt.ion()
        
        if policy_model_name is not None:
            self.load_model(policy_model_name, net_name='policy')
    
        if target_model_name is not None:
            self.load_model(target_model_name, net_name='target')
        
        self.date = datetime.now().strftime("%Y%m%d_%Hh%Mm")
        self.update_cnt = 1

    def load_model(self, model_name, net_name='policy'):
        if net_name == 'policy':
            self.policy_net.load_state_dict(torch.load(model_name, weights_only=True))

        if net_name == 'target':
            self.target_net.load_state_dict(torch.load(model_name, weights_only=True))
    
    def learn(self):
        print(f'START LEARN DQN')

        _memory = ReplayMemory(capacity=self._config.dqn_config.replay_memory_size)
        
        self._same_pos_cnt, self._fail_cnt, self._succeed_cnt, self._over_step_cnt = 0, 0, 0, 0
        _total_step = 0
        
        for i_episode in range(1, self._config.dqn_config.num_episodes+1):
            self._env.reset()

            # 'cur_pos, cur_state, action, next_pos, next_state, reward, done'
            _observation = self.preprocessing(0, None, None, 0, None, 0., False)

            if self._animation_mode:
                _screen = self._env.get_screen() # current_screen : (1, 3, 256, 256)
                _img = self._ax.imshow(_screen)

            _memory_cache = []
            _step = 0

            while True:
                _step, _total_step = _step+1, _total_step+1

                if _step > self._config.dqn_config.limit_step :
                    self._over_step_cnt += 1
                    break

                _action = self._select_action(_total_step, _observation.cur_state)   # {0: 'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}
                _observation = self.step(_observation, _action)
                _observation = self._calculate_reward(_observation)
                _memory_cache.append(_observation)
      
                if self._animation_mode:
                    _screen = self._env.get_screen()
                    _img.set_data(_screen)

                if _observation.done: 
                    break

            _memory = self._memory_push(_memory, _memory_cache)
            self._update_policy_net(_memory)

            print(f'\rNOW EPISODE :{i_episode:>6}, MEMORY : {len(_memory):>6}', end='')

            if i_episode % self._config.dqn_config.target_update == 0:
                self._sync_nets()
                self._evaluate_policy(i_episode=i_episode)
                print(f' >> [{i_episode:>5}/{self._config.dqn_config.num_episodes}] TARGET MODEL UPDATE :  (SAME POS-{self._same_pos_cnt}, FAIL-{self._fail_cnt}, SUCCEED-{self._succeed_cnt}), OVER LIMIT STEP CNT-{self._over_step_cnt}')
        
        print(f'\nCOMPLETE : (SUCCEED-{self._succeed_cnt}/{self._config.dqn_config.num_episodes})')
        self._save_weights()


    def _evaluate_policy(self, i_episode):
        fixed_states = self._get_fixed_states()
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(fixed_states)    # (16, 4)
        q_values_dict = {i: q_values[i].tolist() for i in range(q_values.size(0))}
        render_map(self._ax_eval, q_values_dict, i_episode, self._map_size)
        self.policy_net.train()
        return q_values

    def _get_fixed_states(self):
        hole_layer = self._config.env_config.hole_layer_4
        goal_layer = self._config.env_config.goal_layer_4

        # 캐릭터 가능한 위치 (0~15)
        cur_poses = list(range(16))
        fixed_states = []

        for cur_pos in cur_poses:
            cur_pos_r, cur_pos_c = divmod(cur_pos, self._map_size)
            
            # 캐릭터 위치 레이어 초기화
            character_layer = torch.zeros((self._map_size, self._map_size), dtype=torch.float32)
            character_layer[cur_pos_r, cur_pos_c] = 1.0  # 현재 위치만 1

            # 3채널 (character, hole, goal) stack
            state = torch.stack([character_layer, hole_layer, goal_layer], dim=0)  # shape (3,4,4)
            fixed_states.append(state)

        # 최종 텐서로 변환
        fixed_states = torch.stack(fixed_states, dim=0)  # shape (16, 3, 4, 4)
        return fixed_states
        
    def preprocessing(self, *args):    # *args: cur_pos, cur_state, action, next_pos, next_state, reward, done
        cur_pos, cur_state, action, next_pos, next_state, reward, done = args

        if cur_state is None:
            cur_pos_r, cur_pos_c = divmod(cur_pos, self._map_size)
            if self._map_size == 4:
                cur_state = np.array([
                    [   # 캐릭터 위치
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0]
                    ],
                    [   # 웅덩이 위치
                        [0,0,0,0],
                        [0,1,0,1],
                        [0,0,0,1],
                        [1,0,0,0]
                    ],
                    [   # Goal 위치
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,1]
                    ]
                ])
            cur_state[0][cur_pos_r][cur_pos_c] = 1
            cur_state = torch.tensor(cur_state, dtype=torch.float32, device=self._device)
            cur_state = cur_state.unsqueeze(0)

        if next_pos is not None:
            next_pos_r, next_pos_c = divmod(next_pos, self._map_size)
            if self._map_size == 4:
                next_state = np.array([
                    [   # 캐릭터 위치
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0]
                    ],
                    [   # 웅덩이 위치
                        [0,0,0,0],
                        [0,1,0,1],
                        [0,0,0,1],
                        [1,0,0,0]
                    ],
                    [   # Goal 위치
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,1]
                    ]
                ])
            next_state[0][next_pos_r][next_pos_c] = 1
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self._device)
            next_state = next_state.unsqueeze(0)
        
        return Observation(cur_pos, cur_state, action, next_pos, next_state, reward, done)

    def _select_action(self, step, state):
        sample = random.random()
        eps_threshold = self._config.dqn_config.eps_end + (self._config.dqn_config.eps_start-self._config.dqn_config.eps_end) * math.exp(-1. * step / self._config.dqn_config.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1,1)
                return action.item()
        else:
            return random.randrange(self.n_actions)
        
    def step(self, prev_observation, action):
        next_pos, reward, done, _, _ = self._env.step(action)
        # 'cur_pos, cur_state, action, next_pos, next_state, reward, done'
        observation = self.preprocessing(prev_observation.next_pos, prev_observation.next_state, action, next_pos, None, reward, done)
        return observation

    def _calculate_reward(self, observation):
        _reward = observation.reward

        # PENALTY : 벽으로 이동
        if observation.cur_pos == observation.next_pos:
            _reward = self._config.dqn_config.same_pos_penalty
            self._same_pos_cnt += 1
        
        # PENALTY : 웅덩이에 빠짐
        if observation.done and observation.reward == 0:
            _reward = self._config.dqn_config.negative_reward
            self._fail_cnt += 1

        # REWARD : GOAL
        if observation.done and observation.reward > 0:
            _reward = self._config.dqn_config.positive_reward
            self._succeed_cnt += 1
        
        _reward = torch.tensor([_reward], device=self._device)
        
        return Observation(observation.cur_pos, observation.cur_state, observation.action, observation.next_pos, observation.next_state, _reward, observation.done)
        
    def _memory_push(self, memory, memory_cache):
        for observation in memory_cache:
            memory.push(observation.cur_state, observation.action, observation.next_state, observation.reward)
        return memory

    def _update_policy_net(self, memory):
        if len(memory) < self._config.dqn_config.batch_size * self.update_cnt:
            return
        else:
            self.update_cnt += 1
        
        # 배치 샘플링
        transitions = memory.sample(self._config.dqn_config.batch_size)
        batch = Transition(*zip(*transitions))

        # 상태, 행동, 보상 배치 생성
        state_batch = torch.cat(batch.cur_state)
        action_batch = torch.tensor(batch.action, device=self._device).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)

        # next_state가 존재하는지 마스크 생성
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self._device, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        
        # 현재 상태에서의 Q-value 계산
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # 실제 선택한 액션에 해당하는 Q값만 선택
    
        # 다음 상태에서의 최대 Q-value 계산
        next_state_values = torch.zeros(self._config.dqn_config.batch_size, device=self._device)
        if non_final_next_states_list:
            non_final_next_states = torch.cat(non_final_next_states_list)
            next_q_values = self.target_net(non_final_next_states).max(1)[0]
            next_state_values[non_final_mask] = next_q_values

        # 기대 Q-value 계산
        expected_state_action_values = (next_state_values * self._config.dqn_config.gamma) + reward_batch

        # 손실 함수 계산 (Huber Loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 역전파 및 최적화
        self.optimizer.zero_grad()          # gradient 초기화
        loss.backward()                     # gradient 계산
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)   # gradient clipping
        self.optimizer.step()               # update parameter

    def _sync_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _save_weights(self):
        is_slippery = self._config.env_config.is_slippery
        if is_slippery :
            prefix = 'slippery'
        else:
            prefix = 'not_slippery'
        torch.save(self.policy_net.state_dict(), f"{self._config.dqn_config.weight_path}/{prefix}_policy_model_{self._config.dqn_config.num_episodes}_{self.date}.pt")
        torch.save(self.target_net.state_dict(), f"{self._config.dqn_config.weight_path}/{prefix}_target_model_{self._config.dqn_config.num_episodes}_{self.date}.pt")
    
    def get_action(self, state):
        with torch.no_grad():
            action = self.policy_net(state).max(1)[1].view(1,1)
            return action.item()
        
Transition = namedtuple('Transition', 'cur_state, action, next_state, reward')
class ReplayMemory:
    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)

    def push(self, *args):
        self._memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)
    
    def __len__(self):
        return len(self._memory)

# Q 테이블 렌더링 함수
def render_map(ax, data, i_episode, map_size=4):
    ax.cla()

    ACTION = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}

    if map_size == 4:
        x_positions = [(1, 1), (1, 3), (2, 3), (3, 0)]
        goal_position = (3,3)
    elif map_size == 8:
        x_positions = [(2, 3), (3, 5), (4, 3), (5, 1), (5,2), (5,6), (6,1), (6,4), (6,6), (7,3)]
        goal_position = (7,7)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_title(f'Q-VALUES AFTER {i_episode} EPISODES')

    # 각 셀에 값 표시
    for key, values in data.items():
        # key를 좌표로 변환
        r, c = divmod(key, map_size)
        
        # 특정 좌표에 X 추가
        if (r, c) in x_positions:
            # X를 그리드에 꽉 차게 표시
            size = 0.4  # X 크기
            center_x, center_y = c + 0.5, map_size - r - 0.5
            ax.plot(
                [center_x - size, center_x + size],  # 첫 번째 대각선
                [center_y - size, center_y + size],
                color="blue", linewidth=5
            )
            ax.plot(
                [center_x - size, center_x + size],  # 두 번째 대각선
                [center_y + size, center_y - size],
                color="blue", linewidth=5
            )
        elif (r,c) == goal_position:
            ax.text(
                    c + 0.5, map_size - r - 0.5,  # 텍스트 위치
                    f"GOAL",
                    ha='center', va='center', fontsize=8, color=color, fontweight="bold"
                )
        else:
            # 가장 큰 값과 가장 작은 값 찾기
            max_value = max(values)
            min_value = min(values)


            # 각 방향별로 텍스트 렌더링
            for i, value in enumerate(values):
                # 색상 결정
                color = (
                    "black" if value == 0.00 else
                    "red" if value > 0 and value == max_value else
                    "lightcoral" if value > 0 else
                    "blue" if value < 0 and value == min_value else
                    "cornflowerblue" if value < 0 else
                    "black"
                )
                fontweight = ( "bold" if color == "red" or color == "blue" else
                            "normal")

                # 방향에 따른 텍스트 오프셋 설정
                offsets = {
                    0: (-0.3, 0.0),  # LEFT: 셀 왼쪽
                    1: (0.0, -0.3),  # DOWN: 셀 아래쪽
                    2: (0.3, 0.0),   # RIGHT: 셀 오른쪽
                    3: (0.0, 0.3)    # UP: 셀 위쪽
                }
                offset_x, offset_y = offsets[i]

                # 텍스트 내용 설정
                ax.text(
                    c + 0.5 + offset_x, map_size - r - 0.5 + offset_y,  # 텍스트 위치
                    f"{ACTION[i]}:{value:.2f}",
                    ha='center', va='center', fontsize=10, color=color, fontweight=fontweight
                )

    # 그리드 설정
    ax.set_xticks(np.arange(0, map_size + 1, 1))
    ax.set_yticks(np.arange(0, map_size + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="both", color="black", linestyle='-', linewidth=1)

    # 플롯 표시
    plt.pause(0.01)
    
if __name__ == '__main__':
    from config import Config
    from environment import Environment
    config = Config()
    env = Environment(config.env_config)
    agent = DQN(config=config, env=env, animation_mode=True)
    agent.learn()