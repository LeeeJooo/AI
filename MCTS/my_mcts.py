from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import random
import copy

N = 3   # board size
BLACK_STONE = 1
WHITE_STONE = -1

@dataclass
class Node :
    cum_value: int = 0                                                                  # 누계 가치
    n_action: int = 0                                                                   # 시행 횟수
    board: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))       # 현재 보드 상태
    parent: Optional["Node"] = None                                                     # 부모 노드
    children: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))    # 자식 노드드

    def make_child_node(self):
        child_node = Node(
            board = copy.deepcopy(self.board),
            parent = self
        )
        self.children = np.append(self.children, child_node)

        return child_node
    
    def copy_board(self):
        return copy.deepcopy(self.board)

    def __repr__(self):
        board_str = '\n'.join(
            ' '.join(str(self.board[i][j]) for j in range(self.board.shape[1]))
            for i in range(self.board.shape[0])
        )
        return (f"Node(\n"
                f"value={self.cum_value}, actions={self.n_action},\n"
                f"board=\n"
                f"{board_str},\n"
                f"n_children={len(self.children)},\n"
                f"parent_id={id(self.parent) if self.parent else None}"
                f")\n")

class MCTS() :
    def __init__(self):
        self.a = 0

    def mcts(self, root_node):
        done = False

        while not done:
            # 1. EXPAND
            self.expand(root_node)

            # 2. SELECT
            selected_node = self.select(root_node)

            # 3. EVALUATE
            self._evaluate(self, selected_node)

            # 4. BACKUP
        # return selected_child_node

    # 자식 노드 생성
    def expand(self, parent_node):
        # 빈칸 탐색
        blank_node = []
        for r in range(N):
            for c in range(N):
                if parent_node.board[r][c] == 0:
                    blank_node.append((r,c))

        child_cnt = len(blank_node)
        for i in range(child_cnt-1):
            for j in range(i+1, child_cnt):
                (r1, c1), (r2, c2) = blank_node[i], blank_node[j]
                
                child_node1 = parent_node.make_child_node()
                child_node1.board[r1][c1] = BLACK_STONE   # 흑돌
                child_node1.board[r2][c2] = WHITE_STONE  # 백돌

                child_node2 = parent_node.make_child_node()
                child_node2.board[r2][c2] = BLACK_STONE   # 흑돌
                child_node2.board[r1][c1] = WHITE_STONE  # 백돌

    def select(self, parent_node):
        children_nodes = parent_node.children
        if parent_node.n_action == 0 :
            children_cnt = children_nodes.shape[0]
            selected_idx = random.randrange(0, children_cnt)
            return children_nodes[selected_idx]

        else:
            ubc1 = np.array([children_nodes.cum_value/children_nodes.n_action + ((2+np.log(parent_node.n_action)/children_nodes.n_action)**2)**2 if children_nodes.n_action !=0 else 0])
            selected_idx = np.argmax(ubc1)
            return children_nodes[selected_idx]
    '''def cal_ubc1
        param : numpy.array, dtype=Node >> collection of Child Nodes
        total_n_playout = sum(n_playout of Child Nodes)
        cum_value/n_playout + (2 + log(total_n_playout) /n_playout)**1/2)**1/2
    '''

    def _evaluate(self, start_node):
        cur_board = start_node.copy_board()

        done = False
        while not done:
            # 게임 판정
            done = self._match_dicision(cur_board)

            # 게임 진행행
            cur_board = self._step(cur_board)

        cur_board = self._step(cur_board)

    def _match_dicision(self, board):
        # 승패가 났는지 확인
        

        # 돌 둘 자리가 있는지 확인
        blank_cnt = 0
        for r in range(N):
            for c in range(N):
                if board[r][c] == 0:
                    blank_cnt += 1

        if blank_cnt < 2 : return True


    def _step(self, board):
        # 빈칸 탐색
        blank_node = []
        for r in range(N):
            for c in range(N):
                if board[r][c] == 0:
                    blank_node.append((r,c))
        
        random.shuffle(blank_node)

        pos1, pos2 = blank_node[:2]

        board[pos1[0]][pos1[1]] = BLACK_STONE
        board[pos2[0]][pos2[1]] = WHITE_STONE

        return board

    # def backup

    # def playout

    # 

if __name__ == "__main__":
    init_board = np.zeros(shape=(N, N), dtype=int)  # 게임 초기 값
    root_node = Node(board=init_board)
    mcts = MCTS()
    next_node = mcts.mcts(root_node)