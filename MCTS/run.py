from config import *
from mcts import MCTS
from environment import Environment
from human import Human

def play_game(env, player1, player2):

    end = False
    while not end:
        env.waiting(player1.pid)
        player1_action = player1.get_action()
        env.step(player1.pid, player1_action, visualize=True)
        end, winner = env.end_game()
        if end : break
        env.waiting(player2.pid)
        player2_action = player2.get_action()
        env.step(player2.pid, player2_action, visualize=True)
        end, winner = env.end_game()

    if winner == PIDS[1]:
        print('PLAYER1 WINS')
    if winner == PIDS[2]:
        print('PLAYER2 WINS')
    if winner == None:
        print('DRAW')

if __name__ == "__main__":
    print('START OMOK GAME')

    env = Environment()
    # player1 = MCTS(env, PIDS[1])
    player1 = Human(env, PIDS[1])
    player2 = MCTS(env, PIDS[2])

    play_game(env, player1, player2)