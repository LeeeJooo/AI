import numpy as np
from config import *
from collections import defaultdict
from copy import deepcopy

class Game() :
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.board = self._init_board()
        self.actions = defaultdict()    # key: 좌표(action), value: color(pid)
        self.N_IN_ROW = 5

    def _init_board(self):
        return np.zeros((self.row, self.col))
    
    def current_state(self):
        return self.board
    
    def end_game(self):
        # NOT END : 턴 수 부족
        if len(self.actions) < self.N_IN_ROW*2-1 :
            return False, None

        dirs = {'상하':0, '좌우':1, '좌상우하':2, '우상좌하':3}
        d_ = {'상하':[(-1, 0), (1, 0)], '좌우':[(0, -1),(0, 1)], '좌상우하':[(-1, -1), (1, 1)], '우상좌하':[(-1, 1), (1, -1)]}
        visited = [[[False] * self.col] * self.row] * 4

        for (r, c) in self.actions: # (r, c) : 돌 위치
            for dir, dir_i in dirs.items():
                # PASS: 해당 좌표, 해당 방향으로 이미 탐색 완료.
                if visited[dir_i][r][c] :
                    continue     

                # 탐색 시작
                visited[dir_i][r][c] = True

                cnt = 1
                pid = self.board[r][c]
                for d_r, d_c in d_[dir]:
                    r_, c_ = r, c
                    while True:
                        r_ += d_r
                        c_ += d_c
                        # PASS : 보드 범위 밖. 다른 방향 탐색.
                        if r_<0 or c_<0 or r_>=self.row or c_>=self.col:
                            break
                        
                        if self.board[r_][c_] == pid:
                            cnt += 1
                            visited[dir_i][r_][c_] = True
                        else:
                            break

                
                if cnt >= self.N_IN_ROW:
                    return True, pid
                
        return False, None
    
    def step(self, pid, action):
        self.board[action[0]][action[1]] = pid
        return self.board

    def rollback(self, state):
        self.board = deepcopy(state)


