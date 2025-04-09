from copy import deepcopy
import numpy as np

class Node():
    def __init__(self, state, actions):
        self.state = state  # np.ndarray, shape (ROW, COL)
        self.actions = actions  # set, {action}
        self.action = None  # tuple, (r, c)
        self.value_sum = 0
        self.n_action = 0
        self.parent_node = None # Node
        self.child_node = None  # np.ndarray
        self.winner = 0 # 0-진행중, 1-PLAYER1 WIN, 2-PLAYER2 WIN
    
    def has_child(self):
        if self.child_node is None:
            return False
        else:
            return True

    def make_child_node(self, pid, legal_action):
        node = Node(state=deepcopy(self.state), actions=deepcopy(self.actions))
        node.actions.add(legal_action)
        node.action = legal_action
        node.parent_node = self

        node.state[legal_action[0]][legal_action[1]] = pid

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
    
    def copy_actions(self):
        return deepcopy(self.actions)
    
    def update_value_sum(self, value):
        self.value_sum += value

    def update_n_action(self):
        self.n_action += 1

    def is_root_node(self):
        if self.parent_node == None:
            return True
        else:
            return False
        
    def __str__(self):
        state_str = '[STATE]\n'
        row, col = self.state.shape
        for r in range(row):
            for c in range(col):
                state_str += str(int(self.state[r][c]))
                state_str += ' '
            state_str += '\n'

        actions_str = '[ACTIONS]: {'
        for r, c in self.actions:
            actions_str += '('+str(r)+', '+str(c)+'), '
        actions_str += ' }'

        action_str = '[ACTION] : '
        if self.action is not None:
            action_str += '('+str(self.action[0])+', '+str(self.action[1])+')'
        else:
            action_str += 'NO ACTION'
        value_sum_str = '[VALUE_SUM] : '\
            +str(self.value_sum)
        n_action_str = '[N_ACTION] : '\
            +str(self.n_action)
        parent_node_str = '[PARENT NODE] : '
        if self.parent_node is not None:
            parent_node_str += '\n'
            for r in range(row):
                for c in range(col):
                    parent_node_str += str(int(self.parent_node.state[r][c]))
                    parent_node_str += ' '
                parent_node_str += '\n'
        else: 
            parent_node_str += 'NO PARENT NODE'
        child_node_str = '[CHILD NODE] : '
        if self.child_node is None:
            child_node_str+='No child'
        else:
            child_node_str += str(len(self.child_node))
        
        game_state_str = '[게임 종료 여부] : '
        if self.winner == 0:
            game_state_str += '게임 미종료'
        else:
            game_state_str += '게임 진행중 (PLAYER '
            game_state_str += str(self.winner)
            game_state_str += ' 승)'
        
        return state_str + '\n'\
             + actions_str + '\n'\
             + action_str + '\n'\
             + value_sum_str + '\n'\
             + n_action_str + '\n'\
             + parent_node_str + '\n'\
             + child_node_str + '\n'\
             + game_state_str + '\n'

if __name__ == '__main__':
    board = np.array([
        [1,2,3],
        [2,3,4]
    ])
    board2 = np.array([
        [1,2,3],
        [2,3,4]
    ])
    pn = Node(board2, actions=set({1,2,3}))
    actions = set()
    n = Node(board, actions)
    n.parent_node = pn
    print(n)