import numpy as np
from config import *
import random
from copy import deepcopy
from visualizer import *

class Game() :
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.board = self._init_board()
        self.N_IN_ROW = N_IN_ROW

    def _init_board(self):
        return np.zeros((self.row, self.col))
    
    def current_state(self):
        return self.board
        
    def step(self, pid, action, state=None, visualize=False):
        if state is None:
            state = self.board

        state[action[0]][action[1]] = pid

        if visualize:
            stepVisualizer(pid, action, self.board)
        return state
    
    def wait_visualizer(self, pid):
        waitVisualizer(pid)

    def end_game(self, state=None):
        if state is None:
            state = self.board

        actions = self._get_actions(state)
        
        # NOT END : 턴 수 부족
        if len(actions) < self.N_IN_ROW*2-1 :
            return False, None

        dirs = {'상하':0, '좌우':1, '좌상우하':2, '우상좌하':3}
        d_ = {'상하':[(-1, 0), (1, 0)], '좌우':[(0, -1),(0, 1)], '좌상우하':[(-1, -1), (1, 1)], '우상좌하':[(-1, 1), (1, -1)]}
        visited = [[[False for _ in range(self.col)] for _ in range(self.row)] for _ in range(4)]


        for (r, c) in actions: # (r, c) : 돌 위치
            for dir, dir_i in dirs.items(): # 탐색 방향: 상하/ 좌우/ 좌상우하/ 우상좌하

                # PASS: 해당 좌표, 해당 방향으로 이미 탐색 완료.
                if visited[dir_i][r][c] :
                    continue     

                # 탐색 시작
                visited[dir_i][r][c] = True

                cnt = 1
                pid = state[r][c]
                for d_r, d_c in d_[dir]:
                    r_, c_ = r, c
                    while True:
                        r_ += d_r
                        c_ += d_c
                        # PASS : 보드 범위 밖. 다른 방향 탐색.
                        if r_<0 or c_<0 or r_>=self.row or c_>=self.col:
                            break
                        
                        if state[r_][c_] == pid:
                            cnt += 1
                            visited[dir_i][r_][c_] = True
                        else:   # PASS : 다른 색 돌.
                            break

                
                if cnt >= self.N_IN_ROW:
                    return True, pid
                
        return False, None


    def rollback(self, state):
        self.board = deepcopy(state)

    def _get_actions(self, state=None):
        if state is None:
            state = self.board
        action_pool = []
        for r in range(self.row):
            for c in range(self.col):
                if state[r][c] != 0:
                    action_pool.append((r,c))

        return action_pool
    
    def get_legal_actions(self, state):
        row, col = state.shape
        action_pool = []
        player1_n_turn, player2_n_turn = 0, 0
        for r in range(row):
            for c in range(col):
                if state[r][c] == 0:    # 빈곳
                    action_pool.append((r,c))
                if state[r][c] == PIDS[1]:  # 흑돌
                    player1_n_turn += 1
                if state[r][c] == PIDS[2]:  # 백돌
                    player2_n_turn += 1

        if player1_n_turn <= player2_n_turn:    # 흑돌 차례
            return PIDS[1], action_pool
        else:                                   # 백돌 차례
            return PIDS[2], action_pool
    
    def get_legal_action(self, state):
        now_turn, action = self._get_random_action(state)
        return now_turn, action

    def _get_random_action(self, state):
        row, col = state.shape
        action_pool = []
        player1_n_turn, player2_n_turn = 0, 0
        for r in range(row):
            for c in range(col):
                if state[r][c] == 0:
                    action_pool.append((r,c))
                if state[r][c] == PIDS[1]:
                    player1_n_turn += 1
                if state[r][c] == PIDS[2]:
                    player2_n_turn += 1

        idx = random.randint(0, len(action_pool)-1)

        if player1_n_turn == player2_n_turn:
            return PIDS[1], action_pool[idx]
        else:   # player1_n_turn > player2_n_turn
            return PIDS[2], action_pool[idx]
        
    def check_available_action(self, action):
        if self.board[action[0]][action[1]] != 0:
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