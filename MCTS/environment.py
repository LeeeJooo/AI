class Environment:
    def __init__(self, game):

        self.game = game
        self._play_game()

    def _play_game(self):
        self.game.play_game()

    def current_state(self):
        current_state = self.game.current_state()
        return current_state
    
    def step(self, pid, action):
        result = self.game.step(pid, action)
        ''' game에 action request 후 결과 response 받음'''
        return result

    def end_game(self):
        end, winner = self.game.end_game()
        return end, winner
    
    def rollback(self, state):
        self.game.rollback(state)