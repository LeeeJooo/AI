from config import *
from game import Game

class Environment:
    def __init__(self):
        self._start_game()

    def _start_game(self):
        self._game = Game(ROW, COL)

    def current_state(self):
        current_state, actions = self._game.current_state()
        return current_state, actions
        
    def get_legal_actions(self, actions):
        now_turn, legal_actions = self._game.get_legal_actions(actions)
        return now_turn, legal_actions
    
    def end_game(self, state=None, actions=None):
        end, winner = self._game.end_game(state, actions)
        return end, winner
    
    def get_legal_action(self, actions):
        now_turn, action = self._game.get_legal_action(actions)
        return now_turn, action
    
    def step(self, pid, action, state=None, visualize=False):
        state = self._game.step(pid, action, state, visualize)
        return state

    def waiting(self, pid, n_turn):
        self._game.wait_visualizer(pid, n_turn)

    def check_available_action(self, action):
        is_available = self._game.check_available_action(action)
        return is_available