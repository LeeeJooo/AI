import numpy as np
import matplotlib.pyplot as plt

# 8x8x4 배열 샘플 데이터 생성 (예: 무작위 값)
data = np.random.rand(8, 8, 4)  # 8x8 맵에서 각 방향에 대한 값

# 방향 매핑
ACTION = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}

# Q 테이블 렌더링 함수
def render_map(data):
    nrows, ncols = map(int, map_size.split('x'))

    if nrows == 4:
        x_positions = [(1, 1), (1, 3), (2, 3), (3, 0)]
        goal_position = (3,3)
    elif nrows == 8:
        x_positions = [(2, 3), (3, 5), (4, 3), (5, 1), (5,2), (5,6), (6,1), (6,4), (6,6), (7,3)]
        goal_position = (7,7)

    # 맵 크기에 맞게 플롯 크기 조정
    _, ax = plt.subplots(figsize=(ncols * 1.8, nrows * 1.5))  # 기존 대비 2배 크기

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # 각 셀에 값 표시
    for key, values in data.items():
        # key를 좌표로 변환
        r, c = divmod(key, ncols)
        
        # 특정 좌표에 X 추가
        if (r, c) in x_positions:
            # X를 그리드에 꽉 차게 표시
            size = 0.4  # X 크기
            center_x, center_y = c + 0.5, nrows - r - 0.5
            ax.plot(
                [center_x - size, center_x + size],  # 첫 번째 대각선
                [center_y - size, center_y + size],
                color="blue", linewidth=5
            )
            ax.plot(
                [center_x - size, center_x + size],  # 두 번째 대각선
                [center_y + size, center_y - size],
                color="blue", linewidth=5
            )
        elif (r,c) == goal_position:
            ax.text(
                    c + 0.5, nrows - r - 0.5,  # 텍스트 위치
                    f"GOAL",
                    ha='center', va='center', fontsize=8, color=color, fontweight="bold"
                )
        else:
            # 가장 큰 값과 가장 작은 값 찾기
            max_value = max(values)
            min_value = min(values)


            # 각 방향별로 텍스트 렌더링
            for i, value in enumerate(values):
                # 색상 결정
                color = (
                    "black" if value == 0.00 else
                    "red" if value > 0 and value == max_value else
                    "lightcoral" if value > 0 else
                    "blue" if value < 0 and value == min_value else
                    "cornflowerblue" if value < 0 else
                    "black"
                )
                fontweight = ( "bold" if color == "red" or color == "blue" else
                            "normal")

                # 방향에 따른 텍스트 오프셋 설정
                offsets = {
                    0: (-0.3, 0.0),  # LEFT: 셀 왼쪽
                    1: (0.0, -0.3),  # DOWN: 셀 아래쪽
                    2: (0.3, 0.0),   # RIGHT: 셀 오른쪽
                    3: (0.0, 0.3)    # UP: 셀 위쪽
                }
                offset_x, offset_y = offsets[i]
                # offset_y = 0.2 * (1.5 - i)

                # 텍스트 내용 설정
                ax.text(
                    c + 0.5 + offset_x, nrows - r - 0.5 + offset_y,  # 텍스트 위치
                    f"{ACTION[i]}:{value:.2f}",
                    ha='center', va='center', fontsize=10, color=color, fontweight=fontweight
                )

    # 그리드 설정
    ax.set_xticks(np.arange(0, ncols + 1, 1))
    ax.set_yticks(np.arange(0, nrows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="both", color="black", linestyle='-', linewidth=1)

    # 플롯 표시
    plt.show()

import gym
from gym.envs.registration import register
from collections import deque
from collections import defaultdict
import numpy as np
from agent import Agent

# is_slippery = input("is_slippery no or yes : ")
# if is_slippery == "yes":
#     is_slippery = True
# else:
#     is_slippery = False

# map_size = input("map_size 4x4 or 8x8 : ")

is_slippery=True
# is_slippery=False
map_size = '4x4'
# map_size = '8x8'

register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : map_size, 'is_slippery': is_slippery}
    )

env = gym.make('FrozenLake-v3')
action_size = env.action_space.n

def check_environement():
    env.reset()
    env.render()
    print()
    while True:
        action = input("Enter action: ")
        if action not in ['0','1','2','3']:
            continue
        action = int(action)
        state, reward, done, info, _prob_dict = env.step(action)
        env.render()
        print("State :", state, "Action: ", action, "Reward: ", reward, "info: ", info)
        print()
        if done:
            print("Finished with reward", reward)
            break


def testing_after_learning(Q):
    agent = Agent(Q, env, "testing_mode")
    total_test_episode = 1000
    rewards = []
    for episode in range(total_test_episode):
        state = env.reset()
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            new_state, reward, done, info, _prob_dict = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                break
            state = new_state
    print("avg: " + str(sum(rewards) / total_test_episode))

Q = defaultdict(lambda: np.zeros(action_size))
while True:
    print()
    print("1. Checking Frozen_Lake")
    print("2. Q-learning")
    print("3. Testing after learning")
    print("4. Exit")
    menu = int(input("select: "))
    if menu == 1:
        check_environement()
    elif menu == 2:
        Q = defaultdict(lambda: np.zeros(action_size))
        agent = Agent(Q, env, "learning_mode")
        agent.learn()
        render_map(Q)
    elif menu == 3:
        testing_after_learning(Q)
    elif menu == 4:
        break
    else:
        print("wrong input!")

