from game import Game
from config import *

class Environment:
    def __init__(self):
        self._start_game()

    def _start_game(self):
        self._game = Game(ROW, COL)

    def current_state(self):
        current_state = self._game.current_state()
        return current_state
    
    def step(self, pid, action, state=None, visualize=False):
        state = self._game.step(pid, action, state, visualize)
        return state

    def end_game(self, state=None):
        end, winner = self._game.end_game(state)
        return end, winner
    
    def rollback(self, state):
        self._game.rollback(state)

    def get_legal_action(self, state):
        now_turn, action = self._game.get_legal_action(state)
        return now_turn, action
    
    def get_legal_actions(self, state):
        now_turn, next_actions = self._game.get_legal_actions(state)
        return now_turn, next_actions
    
    def waiting(self, pid):
        self._game.wait_visualizer(pid)

    def check_available_action(self, action):
        is_available = self._game.check_available_action(action)
        return is_available