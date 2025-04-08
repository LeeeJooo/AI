from copy import deepcopy
import numpy as np
from collections import defaultdict

class Node():
    def __init__(self, state):
        self.state = state  # np.ndarray, shape (ROW, COL)
        self.action = None  # tuple, (r, c)
        self.value_sum = 0
        self.n_action = 0
        self.parent_node = None
        self.child_node = None
        self.winner = 0
    
    def has_child(self):
        if self.child_node is None:
            return False
        else:
            return True

    def make_child_node(self, pid, next_action):
        node = Node(state=deepcopy(self.state))
        node.action = next_action
        node.parent_node = self

        node.state[next_action[0]][next_action[1]] = pid

        if self.child_node is None :
            self.child_node = np.array([node])
        else:
            self.child_node = np.append(self.child_node, node)

        return node

    def copy_node(self):
        return Node(
            state=deepcopy(self.state),
            value_sum=self.value_sum,
            n_action=self.n_action,
            action=self.action,
            parent_node=deepcopy(self.parent_node),
            child_node=deepcopy(self.child_node),
        )

    def copy_state(self):
        return deepcopy(self.state)
    
    def update_value_sum(self, value):
        self.value_sum += value

    def update_n_action(self):
        self.n_action += 1

    def is_root_node(self):
        if self.parent_node == None:
            return True
        else:
            return False