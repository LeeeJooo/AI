import numpy as np
from config import *
from node import Node
from environment import Environment

class MCTS:
    def __init__(self,env, pid):
        self._env = env
        self.pid = pid

    def get_action(self):
        root_state = self._env.current_state()

        root_node = Node(root_state)
        node_pool = np.array([root_node])
        
        has_enough_trials = False
        n_trial = 0
        while not has_enough_trials and n_trial < LIMIT_N_TRIAL:
            n_trial += 1
            selected_node = self._select(node_pool)

            winner = selected_node.winner
            if winner == 0:
                if not selected_node.has_child():
                    node_pool = self._expand(node_pool, selected_node)
                winner = self._simulate(selected_node)

            node_value = 1 if self.pid==winner else -1
            self._backpropagate(selected_node, node_value)

            has_enough_trials = self._check_n_action(root_node)

        action = self._select_final_action(root_node)
        return action

    def _select_final_action(self, root_node):
        UCB1_scores = self._get_ucb1_scores(root_node.child_node)
        idx = np.argmax(UCB1_scores)
        return root_node.child_node[idx].action
    
    def _select(self, node_pool):
        UCB1_scores = self._get_ucb1_scores(node_pool)
        idx = np.argmax(UCB1_scores)
        return node_pool[idx]

    def _get_ucb1_scores(self, node_pool, c=1.4):
        parent_visits = np.array([node.parent_node.n_action if not node.is_root_node() else node.n_action for node in node_pool])
        visits = np.array([node.n_action for node in node_pool])
        values = np.array([node.value_sum for node in node_pool])

        # 방문 안 한 노드는 inf, 방문한 노드는 UCB1 계산
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_values = values / visits
            # exploration = c * np.sqrt(np.log(parent_visits) / visits)
            exploration = np.sqrt(np.sqrt(np.log(parent_visits) / visits))
            scores = avg_values + exploration
            scores[visits == 0] = float('inf')  # 방문하지 않은 노드는 무조건 선택

        return scores
    
    def _expand(self, node_pool, parent_node):
        # 이미 게임 종료된 상태
        if parent_node.winner != 0:
            return node_pool

        now_turn, next_actions = self._get_legal_actions(parent_node.state)

        for next_action in next_actions:
            child_node = parent_node.make_child_node(now_turn, next_action)
            end, winner = self._end_game(child_node.state)
            if end:
                child_node.winner = winner
            node_pool = np.append(node_pool, child_node)
        
        return node_pool
    
    def _simulate(self, selected_node):
        end=False
        state = selected_node.copy_state()
        while not end:
            now_turn, action = self._get_legal_action(state)
            state = self._step(now_turn, action, state)
            end, winner = self._end_game(state)
        return winner
    
    def _get_legal_action(self, state):
        now_turn, action = self._env.get_legal_action(state)
        return now_turn, action
    
    def _get_legal_actions(self, state):
        now_turn, next_actions = self._env.get_legal_actions(state)
        return now_turn, next_actions

    def _check_n_action(self, root_node):
        if root_node.n_action < N_ACTION:
            return False
        
        for child_node in root_node.child_node:
            if child_node.n_action < N_ACTION:
                return False
            
        return True

    def _step(self, pid, action, state):
        state = self._env.step(pid, action, state)
        return state

    def _end_game(self, state):
        end, winner = self._env.end_game(state)
        return end, winner
    
    def _backpropagate(self, node, node_value):
        if not node.is_root_node():
            self._backpropagate(node.parent_node, node_value)
        
        node.update_value_sum(node_value)
        node.update_n_action()

    def _rollback(self, state):
        self._env.rollback(state)

if __name__ == "__main__":
    env = Environment()
    mcts = MCTS(1, env)
    action = mcts.get_action()