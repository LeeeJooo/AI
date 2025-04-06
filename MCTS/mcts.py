import numpy as np
from config import *
from node import Node
from environment import Environment
import random
from omok import Omok

class MCTS:
    def __init__(self, pid, env, policy):
        self.pid = pid
        self.env = env
        self.policy = policy

    def get_action(self):
        root_state = self.env.current_state()

        root_node = Node(root_state)
        node_pool = np.array([root_node])
        
        has_enough_trials = False
        while not has_enough_trials:
            selected_node = self._select(node_pool)

            if not selected_node.has_child():
                self._expand(node_pool, selected_node)

            end=False
            state = selected_node.copy_state()
            while not end:
                action = self.policy.get_legal_action(state)
                state = self._step(self.pid, action)
                end, winner = self._end_game()
            
            node_value = 1 if self.pid==winner else -1
            self._backpropagate(selected_node, node_value)
            self._rollback(root_state)
            has_enough_trials = self._check_node_n_action(node_pool)

        action_node = self._select(node_pool)
        return action_node.get_action()

    def _select(self, node_pool):
        UCB1_scores = self._get_ucb1_scores(node_pool)
        idx = np.argmax(UCB1_scores)
        return node_pool[idx]

    def _get_ucb1_scores(self, node_pool, c=np.sqrt(2)):
        parent_visits = np.array([node.parent_node.n_action if not node.is_root_node() else 1 for node in node_pool])
        visits = np.array([node.n_action for node in node_pool])
        values = np.array([node.value_sum for node in node_pool])

        # 방문 안 한 노드는 inf, 방문한 노드는 UCB1 계산
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_values = values / visits
            exploration = c * np.sqrt(np.log(parent_visits) / visits)
            scores = avg_values + exploration
            scores[visits == 0] = float('inf')  # 방문하지 않은 노드는 무조건 선택

        return scores
    
    def _expand(self, node_pool, parent_node):
        state = parent_node.state
        next_actions = self.policy.get_legal_actions(state) # List

        for next_action in next_actions:
            child_node = parent_node.make_child_node(self.pid, next_action)
            node_pool = np.append(node_pool, child_node)

    def _check_node_n_action(self, node_pool):
        return all(node.n_action >= N_ACTION for node in node_pool)

    def _step(self, pid, action):
        state = self.env.step(pid, action)
        return state

    def _end_game(self):
        end, winner = self.env.end_game()
        return end, winner
    
    def _backpropagate(self, node, node_value):
        if not node.is_root_node():
            self._backpropagate(node.parent_node, node_value)
        
        node.update_value_sum(node_value)
        node.update_n_action()

    def _rollback(self, state):
        self.env.rollback(state)

if __name__ == "__main__":
    env = Environment()
    policy = Omok()
    mcts = MCTS(1, env, policy)
    action = mcts.get_action()
    print('good')