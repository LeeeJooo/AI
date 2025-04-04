import numpy as np
from config import *
from node import Node
import random

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
            while end:
                action = self.policy.get_next_action(state)
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

    def _get_ucb1_scores(self, node_pool):
        UCB1_func = np.vectorize(lambda node: node.value_sum/node.n_action + ((2*np.log(node.parent.n_action)/node.value_sum)**1/2)**1/2)
        return UCB1_func(node_pool)

    def _expand(self, node_pool, parent_node):
        state = parent_node.get_state()
        next_actions = self.policy.get_all_next_actions(state)

        for next_action in next_actions:
            child_node = parent_node.make_child_node(self.pid, next_action)
            node_pool = np.append(node_pool, child_node)

    def _check_node_n_action(self, node_pool):
        return all(node.n_action >= N_ACTION for node in node_pool)

    def _step(self, action):
        state = self.env.step(self.pid, action)
        return state

    def _end_game(self, state):
        end, winner = self.env.end_game()
        return end, winner
    
    def _backpropagate(self, node, node_value):
        if not node.is_root_node():
            parent_node = node.get_parent()
            self._backpropagate(parent_node, node_value)
        
        node.update_value_sum(node_value)
        node.update_n_action()

    def _rollback(self, state):
        self.env.rollback(state)