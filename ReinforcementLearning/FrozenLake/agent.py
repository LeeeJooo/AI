import numpy as np

ACTION = {0: 'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}

class Agent:
    def __init__(self, Q, env, mode):
        self._Q = Q
        self._env = env
        self._debug = False
        if mode == 'learning_mode': self._debug = True
        

    '''
    q 업데이트 식 오류 정리
    epsilon 차이 (없을수도)
    '''
    def learn(self):
        np.random.seed(42)
        
        MAX_EPISODE = 100
        
        POS_REWARD = 1
        NEG_REWARD = -1

        _alpha = 0.5
        ALPHA_MIN = 0.1
        ALPHA_DECAY = 0.999

        _epsilon = 1.0
        EPSILON_MIN = 0.1
        EPSILON_DECAY = 0.99
        
        _gamma = 0.5
        GAMMA_MAX = 0.9
        GAMMA_INCRE = 1.01

        _action_size = self._env.action_space.n
        _avg_reward = 0

        for episode in range(1, MAX_EPISODE+1):
            _state = self._env.reset()[0]

            if self._debug: self._env.render()
 
            _episode_reward = 0

            while True:
                # 1) action 선택
                if np.random.random() <= _epsilon:
                    _action = np.random.randint(0, _action_size-1) # Exploration
                else:
                    # [방법1] ε-greedy 정책
                    _action_indices = np.where(self._Q[_state] == max(self._Q[_state]))[0]
                    _action = np.random.choice(_action_indices)

                    # [방법2] Boltzmann 정책
                    # _probabilities = np.exp(self._Q[_state]) / np.sum(np.exp(self._Q[_state]))
                    # _action = np.random.choice(range(_action_size), p=_probabilities)

                # 2) action 수행 
                _new_state, _reward, _done, _, _ = self._env.step(_action)
                
                # 3) 보상 설정
                if _done and _reward > 0 : _reward = POS_REWARD   # 도착
                elif _done : _reward = NEG_REWARD                 # 웅덩이에 빠짐

                # 4) Q 값 업데이트
                self._Q[_state][_action] = self._Q[_state][_action] + _alpha * (_reward + _gamma * max(self._Q[_new_state]) - self._Q[_state][_action])
                
                # 상태와 보상 업데이트
                _state = _new_state
                _episode_reward += _reward
                
                # 5) EPISODE 종료
                if _done:
                    # 현재 episode의 평균 reward
                    _avg_reward += (_episode_reward - _avg_reward)/episode
                    if episode%100==0:
                        print("\rEpisode {}/{} || average reward {}".format(episode, MAX_EPISODE, _avg_reward), end="") 
                    break
            
            if _epsilon > EPSILON_MIN: _epsilon *= EPSILON_DECAY
            if _alpha > ALPHA_MIN: _alpha *= ALPHA_DECAY
            if _gamma < GAMMA_MAX: _gamma *= GAMMA_INCRE
            
        
    def select_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        _action_indices = np.where(self._Q[state] == max(self._Q[state]))[0]
        _action = np.random.choice(_action_indices)
        return _action