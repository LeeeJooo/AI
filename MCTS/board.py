import numpy as np

class Board():
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def init_board(self):
        return np.zeros((self.height, self.width))
    
    def step(self, pid, board_state, action):
        pass

    def end_game(self, board_state):
        pass

if __name__ == "__main__":
    board = Board(2, 3)
    b = board.init_board()
    print(b)
