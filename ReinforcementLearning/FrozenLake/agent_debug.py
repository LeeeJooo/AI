import random
import numpy as np
import matplotlib.pyplot as plt

ACTION = {0: 'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}

class Agent:
    def __init__(self, Q, env, mode):
        self._Q = Q
        self._env = env
        self._debug = False
        if mode == 'learning_mode': self._debug = True
        
    def learn(self):
        # POS 보상을 더 크게
        POS_REWARD = 1
        NEG_REWARD = -100

        _alpha = 0.0001
        _gamma = 0.5
        _epsilon = 0.9
        _max_episode = 100000

        min_avg, max_avg = 10e9, -10e9
        _action_size = self._env.action_space.n
        _nrow, _ = self._env.unwrapped.desc.shape

        _succeed_cnt = 0

        result = []
        _episode_avg_reward = 0


        for episode in range(1, _max_episode+1):
            _state = self._env.reset()[0]

            self._env.render()
            _step_cnt, _exploration_cnt, _exploitation_cnt, _step_avg_reward = 0, 0, 0, 0
            _is_succeeded = False
            # if self._debug : print(f'\n[ Episode {episode}/{_max_episode} ]')
            while True:
                _before_q_values = np.copy(self._Q[_state])
                _step_cnt += 1
                
                # 1) action 선택
                # EXPLORATION
                if random.random() <= _epsilon:
                    _action_type = '탐험'
                    _exploration_cnt += 1
                    _action = random.randint(0, _action_size-1)
                # EXPLOITATION
                else:
                    _action_type = '활용'
                    _exploitation_cnt += 1
                    _action_indices = np.where(self._Q[_state] == max(self._Q[_state]))[0]
                    _action = np.random.choice(_action_indices)

                # 2) action 수행 
                _new_state, _reward, _done, _, _ = self._env.step(_action)
                self._env.render()

                # 3) reward 설정
                if _state == _new_state:    # ? model free에 위반 ?
                    _reward = NEG_REWARD
                if _done and _reward==0.0 :
                    _is_succeeded = False
                    _reward = NEG_REWARD
                elif _done and _reward>0.0 :
                    _succeed_cnt += 1
                    _is_succeeded = True
                    _reward = POS_REWARD

                # 현재 episode의 평균 reward
                _step_avg_reward += (_reward - _step_avg_reward)/_step_cnt

                _before_q_value = np.copy(self._Q[_state][_action])

                # 4) Q 값 업데이트
                # (1-a)q + a * (r + g * q' - q)
                self._Q[_state][_action] = (1-_alpha)*self._Q[_state][_action] + _alpha * (_reward + _gamma * max(self._Q[_new_state]) - self._Q[_state][_action])

                # debug
                # self.print_step_result(_step_cnt, _before_q_values, _action_type, _state, _nrow, _action, _new_state,_reward, _step_avg_reward, _before_q_value):
                
                # 5) state 업데이트
                _state = _new_state

                # EPISODE 종료
                if _done:
                    _episode_avg_reward += (_step_avg_reward - _episode_avg_reward)/episode
                    min_avg = min(min_avg, _step_avg_reward)
                    max_avg = max(max_avg, _step_avg_reward)
                    result.append(_succeed_cnt)
                    if self._debug and episode % 100 == 0:  # debug
                        self.print_episode_result(episode, _max_episode, _step_cnt, _exploration_cnt, _exploitation_cnt, _is_succeeded, _succeed_cnt, _step_avg_reward, _episode_avg_reward, max_avg, min_avg)
                    break
        
        if self._debug :self.show_result(result)

    def select_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        _action_indices = np.where(self._Q[state] == max(self._Q[state]))[0]
        _action = np.random.choice(_action_indices)
        return _action

    def print_step_result(self, _step_cnt, _before_q_values, _action_type, _state, _nrow, _action, _new_state,_reward, _step_avg_reward, _before_q_value):
        print(f'[STEP {_step_cnt}]')
        print('   [Q-values]', end=' ')
        for k, dir in ACTION.items():
            print(f'{dir} : {_before_q_values[k]: .2f}', end=', ')
        print()
        print(f'   [{_action_type}]', end=' ')                         # 탐험/활용
        _now_row, _now_col = _state//_nrow, _state%_nrow
        print(f'BEFORE : ({_now_row},{_now_col})', end=' >> ')      # _state
        print(F'{ACTION[_action]} 으로 이동', end=' >> ')   # _action
        _next_row, _next_col = _new_state//_nrow, _new_state%_nrow
        print(f' AFTER : ({_next_row},{_next_col})')    # _new_state
        print(f'   [REWARD] : now_reward: {_reward}, step_avg_reward: {_step_avg_reward}')  # _reward
        print(f'   [{_state}][{_action}] Q-value>>  BEFORE : {_before_q_value: .2f} >> AFTER : {self._Q[_state][_action]: .2f}')
    
    def print_episode_result(self, episode, _max_episode, _step_cnt, _exploration_cnt, _exploitation_cnt, _is_succeeded, _succeed_cnt, _step_avg_reward, _episode_avg_reward, max_avg, min_avg):
        print(f'\n[ Episode {episode}/{_max_episode} ]')
        print(f'>> STEP: {_step_cnt}, 탐험: {_exploration_cnt}, 활용: {_exploitation_cnt}')
        print(f'>> Episode {episode}/{_max_episode} : ', end='')
        print('성공' if _is_succeeded else '실패', end=' ')
        print(f'(성공: {_succeed_cnt}, 실패: {episode-_succeed_cnt})')
        print(f'>> now episode avg reward : {_step_avg_reward: .2f}')
        print(f'>> episode avg reward : {_episode_avg_reward: .2f}')
        print(f'>> MAX Avg : {max_avg: .2f}, MIN Avg : {min_avg: .2f}')

    def show_result(self, result):
        x = range(len(result))

        plt.figure(figsize=(10, 6))
        plt.plot(x, result, marker='.', label='is_succeeded', linewidth=1)
        
        # 마지막 값 표시
        last_x = len(result) - 1
        last_y = result[-1]
        plt.text(last_x, last_y, f'{last_y}', fontsize=12, fontweight='bold', ha='center', va='bottom', color='red')  # 값 표시

        plt.xlabel("episode", fontsize=12)
        plt.xlabel("success", fontsize=12)
        plt.grid(alpha=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()