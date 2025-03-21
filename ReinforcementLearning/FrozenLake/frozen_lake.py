import gym
import numpy as np
from agent import Agent
from collections import defaultdict
from gym.envs.registration import register

# is_slippery = input("is_slippery no or yes : ")
# if is_slippery == "yes":
#     is_slippery = True
# else:
#     is_slippery = False

# map_size = input("map_size 4x4 or 8x8 : ")

is_slippery = True
# map_size = '4x4'
map_size = '8x8'

register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : map_size, 'is_slippery': is_slippery}
    )

env = gym.make('FrozenLake-v3')
# env = gym.make('FrozenLake-v3', render_mode = 'human')
action_size = env.action_space.n
_nrow, _ = env.unwrapped.desc.shape
ACTION = {0: 'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}

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
    # env = gym.make('FrozenLake-v3', render_mode="human")
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
# agent = Agent(Q, env, "learning_mode")
agent = Agent(Q, env, "testing_mode")
agent.learn()


print()
q_list = [None] * (max(Q.keys()) + 1)
for k, v in Q.items():
    q_list[k] = v

for idx, q_values in enumerate(q_list):
    if q_values is not None:  # Q 값이 존재하는 경우만 출력
        r, c = idx // _nrow, idx % _nrow
        print(f'({r},{c}) >> ', end='')
        for i, q in enumerate(q_values):
            print(f'{ACTION[i]}: {q: .4f}, ', end='')
        print('\r')

testing_after_learning(Q)

# while True:
#     print()
#     print("1. Checking Frozen_Lake")
#     print("2. Q-learning")
#     print("3. Testingn after learning")
#     print("4. Exit")
#     menu = int(input("select: "))
#     if menu == 1:
#         check_environement()
#     elif menu == 2: 
#         # env = gym.make('FrozenLake-v3', render_mode="rgb_array")
#         Q = defaultdict(lambda: np.zeros(action_size))
#         agent = Agent(Q, env, "learning_mode")
#         # agent = Agent(Q, env, "testing_mode")
#         agent.learn()
#     elif menu == 3:
#         testing_after_learning(Q)
#     elif menu == 4:
#         break
#     else:
#         print("wrong input!")