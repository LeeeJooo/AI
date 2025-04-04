from config import *
from mcts import MCTS
from game import Game
from environment import Environment

def play_game(env, player1, player2):

    end = False
    while not end:
        player1_action = player1.get_action()
        env.step(player1_action)
        end, winner = env.end_game()
        if end : break
        player2_action = player2.get_action()
        env.step(player2_action)
        end, winner = env.end_game()

    if winner == PIDS[1]:
        print('PLAYER1 WINS')
    if winner == PIDS[2]:
        print('PLAYER2 WINS')
    if winner == None:
        print('DRAW')


    pass

if __name__ == "__main__":
    print('START OMOK GAME')

    game = Game(ROW, COL)
    env = Environment(game)
    player1 = MCTS(env, PIDS[0])
    player2 = MCTS(env, PIDS[1])

    play_game(env, player1, player2)