from config import *
from mcts import MCTS
from game import Game
from environment import Environment

def play_game(env, player1, player2):

    end = False
    while not end:
        player1_action = player1.get_action()
        result = env.step(player1_action)

    pass

if __name__ == "__main__":
    print('START OMOK GAME')

    game = Game(ROW, COL)
    env = Environment(game)
    player1 = MCTS(env, PIDS[0])
    player2 = MCTS(env, PIDS[1])

    play_game(env, player1, player2)