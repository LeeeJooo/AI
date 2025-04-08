from config import *
import time
pid_info = {1:'PLAYER 1', 2:'PLAYER 2'}
stone_info = {1:'흑돌', 2:'백돌'}

def stepVisualizer(pid, action=None, board=None):
    print(f'{stone_info[pid]} ({action[0]}, {action[1]})')
    for r in range(ROW+1):
        for c in range(COL+1):
            if r==0:
                if c==0:print('X', end=' ')
                else: print(c-1, end=' ')
            if c == 0 and r!=0:
                print(r-1, end=' ')

            if r > ROW or c > COL: continue
            if r!=0 and c!=0 and board[r-1][c-1] == 0:
                print('_', end=' ')
            if r!=0 and c!=0 and board[r-1][c-1] == 1:
                print('○', end=' ')
            if r!=0 and c!=0 and board[r-1][c-1] == 2:
                print('●', end=' ')
        print()
    # time.sleep(3)

def waitVisualizer(pid):
    print(f'\n현재 턴 : {pid_info[pid]}',end =' >> ')