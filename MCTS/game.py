import random
import numpy as np
from config import *
from visualizer import *

class Game() :
    def __init__(self, r, c):
        self._row = ROW
        self._col = COL
        self._n_in_row = N_IN_ROW
        self._initialize()

    def _initialize(self):
        print('INITIALIZE GAME')
        self._board = self._init_board()
        self._actions = set()
        self._total_actions = self._get_total_actions()

    def _init_board(self):
        return np.zeros((self._row, self._col))
    
    def _get_total_actions(self):
        total_actions = set()
        for r in range(self._row):
            for c in range(self._col):
                total_actions.add((r, c))
        return total_actions

    def current_state(self):
        return self._board, self._actions
    
    def get_legal_actions(self, actions):
        legal_actions = list(self._total_actions - actions)
        actions_cnt = len(actions)

        if actions_cnt%2 == 0:
            return PIDS[1], legal_actions    # 흑돌 턴, 가능한 ACTIONS LIST
        if actions_cnt%2 == 1:
            return PIDS[2], legal_actions    # 백돌 턴, 가능한 ACTIONS LIST
    
    def end_game(self, state=None, actions=None):
        if state is None:
            state = self._board
        if actions is None:
            actions = self._actions
        # NOT END : 턴 수 부족
        if len(actions) < self._n_in_row*2-1 :
            return False, None

        directions = [(-1, 0), (0, -1), (-1, -1), (-1, 1)]  # 수직, 수평, 대각선-좌상우하, 대각선-우상좌하

        for r, c in actions: # (r, c) : 돌 위치
            pid = state[r][c]
            for dr, dc in directions:
                count = 1
                for step in [1, -1]:    # 양방향 탐색
                    nr, nc = r, c
                    while True:
                        nr += dr * step
                        nc += dc * step
                        if not (0 <= nr < self._row and 0 <= nc < self._col):
                            break
                        if state[nr][nc] == pid:
                            count += 1
                        else:
                            break
                if count >= self._n_in_row:
                    return True, pid
        return False, None
    
    def get_legal_action(self, actions):
        now_turn, action = self._get_random_action(actions)
        return now_turn, action
    
    def _get_random_action(self, actions):
        legal_actions = list(self._total_actions - actions)
        
        # DRAW
        if len(legal_actions) == 0:
            return 0, None
        
        random.shuffle(legal_actions)
        actions_cnt = len(actions)

        if actions_cnt%2 == 0:
            return PIDS[1], legal_actions[0]
        if actions_cnt%2 == 1:
            return PIDS[2], legal_actions[0]
        
    def step(self, pid, action, state=None, visualize=False):
        if state is None:
            state = self._board
            self._actions.add(action)

        state[action[0]][action[1]] = pid

        if visualize:
            stepVisualizer(pid, action, self._board)
        return state
    
    def wait_visualizer(self, pid, n_turn):
        waitVisualizer(pid, n_turn)


    def check_available_action(self, action):
        if self._board[action[0]][action[1]] != 0:
            return False
        else:
            return True

if __name__ == "__main__":
    game1 = Game(3, 3)
    game1.N_IN_ROW = 2
    board = np.array([
        [0,1,0],
        [1,0,0],
        [2,0,0]
    ])
    # board = game1.step(2, (0,1), board)
    end, winner = game1.end_game(board)
    print(f'{end}, {winner}')


    game2 = Game(7, 7)
    game2.N_IN_ROW = 5
    board = np.array([
        [0,2,0,0,0,0,0],
        [2,2,0,0,0,0,2],
        [0,0,2,2,0,0,2],
        [0,0,1,1,1,1,1],
        [0,2,1,1,0,0,0],
        [0,2,1,0,1,0,0],
        [0,2,1,0,1,0,0],
    ])
    end, winner = game2.end_game(board)
    print(f'{end}, {winner}')