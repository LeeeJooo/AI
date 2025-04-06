import random

class Omok:
    def __init__(self):
        pass

    def get_legal_actions(self, state):
        row, col = state.shape
        action_pool = []
        for r in range(row):
            for c in range(col):
                if state[r][c] == 0:
                    action_pool.append((r,c))
        return action_pool
    
    def get_legal_action(self, state):
        action = self._get_random_action(state)
        return action

    def _get_random_action(self, state):
        row, col = state.shape
        action_pool = []
        for r in range(row):
            for c in range(col):
                if state[r][c] == 0:
                    action_pool.append((r,c))
        idx = random.randint(0, len(action_pool)-1)
        return action_pool[idx]
        
