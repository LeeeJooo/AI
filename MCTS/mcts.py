import numpy as np
from config import *
from node import Node
from environment import Environment

# DEPTH 제한
class MCTS:
    def __init__(self,env, pid):
        self._env = env
        self.pid = pid

    def get_action(self):
        root_state, actions = self._env.current_state()

        root_node = Node(root_state, actions)
        node_pool = np.array([root_node])
        
        has_enough_trials = False
        n_trial = 0
        while not has_enough_trials:
            n_trial += 1
            selected_node = self._select(node_pool)

            winner = selected_node.winner
            if winner == 0: # selected node의 state가 게임이 끝나지 않은 상태인 경우
                if not selected_node.has_child():
                    node_pool = self._expand(node_pool, selected_node)
                winner = self._simulate(selected_node)

            # 중간에 REWARD?
            node_value = 1 if self.pid==winner else -1
            self._backpropagate(selected_node, node_value)

            has_enough_trials = self._check_enough_trial(n_trial, root_node)

        action = self._select_final_action(root_node)
        return action

    
    def _select(self, node_pool):
        UCB1_scores = self._get_ucb1_scores(node_pool)
        idx = np.argmax(UCB1_scores)
        return node_pool[idx]

    def _get_ucb1_scores(self, node_pool):
        parent_visits = np.array([node.parent_node.n_action if not node.is_root_node() else node.n_action for node in node_pool])
        visits = np.array([node.n_action for node in node_pool])
        values = np.array([node.value_sum for node in node_pool])

        # 방문 안 한 노드는 inf, 방문한 노드는 UCB1 계산
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_values = values / visits
            exploration = np.sqrt(np.sqrt(np.log(parent_visits) / visits))
            scores = avg_values + exploration
            scores[visits == 0] = float('inf')  # 방문하지 않은 노드는 무조건 선택

        return scores
    
    def _expand(self, node_pool, parent_node):
        now_turn, legal_actions = self._get_legal_actions(parent_node.actions)

        for legal_action in legal_actions:
            child_node = parent_node.make_child_node(now_turn, legal_action)
            end, winner = self._end_game(child_node.state, child_node.actions)
            if end:
                child_node.winner = winner
            node_pool = np.append(node_pool, child_node)
        
        return node_pool
        
    def _get_legal_actions(self, actions):
        now_turn, legal_actions = self._env.get_legal_actions(actions)
        return now_turn, legal_actions
    
    def _end_game(self, state, actions):
        end, winner = self._env.end_game(state, actions)
        return end, winner
    
    def _simulate(self, selected_node):
        end=False
        state = selected_node.copy_state()
        actions = selected_node.copy_actions()
        while not end:
            now_turn, action = self._get_legal_action(actions)
            if now_turn == 0:
                return 0
            state = self._step(now_turn, action, state)
            actions.add(action)
            end, winner = self._end_game(state, actions)
        return winner
    
    def _get_legal_action(self, actions):
        now_turn, action = self._env.get_legal_action(actions)
        return now_turn, action

    def _step(self, pid, action, state):
        state = self._env.step(pid, action, state)
        return state
    
    def _backpropagate(self, node, node_value):
        if not node.is_root_node():
            self._backpropagate(node.parent_node, node_value)
        
        node.update_value_sum(node_value)
        node.update_n_action()

    def _check_enough_trial(self, n_trial, root_node):
        if  n_trial >= LIMIT_N_TRIAL:
            return True
        
        if root_node.n_action < N_ACTION:
            return False
        
        if self._check_child_node_n_action2(root_node):
            return True
        
        return False

    def _check_child_node_n_action2(self, root_node):
        child_node_n_action_sum = sum([node.n_action for node in root_node.child_node])
        child_node_n_action_avg = child_node_n_action_sum / len(root_node.child_node)
        
        if child_node_n_action_avg > N_ACTION:
            return True
            
        return False

    def _check_child_node_n_action1(self, root_node):
        for child_node in root_node.child_node:
            if child_node.n_action < N_ACTION:
                return False
            
        return True

    def _select_final_action(self, root_node):
        # EXPLORATION 값은 제외
        UCB1_scores = self._get_ucb1_scores(root_node.child_node)
        idx = np.argmax(UCB1_scores)
        return root_node.child_node[idx].action

if __name__ == "__main__":
    env = Environment()
    mcts = MCTS(1, env)
    action = mcts.get_action()