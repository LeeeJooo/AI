import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment
from dqn_v1 import DQN
from config import Config
import time

ACTIONS = {0: 'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}

def evaluate_dqn(env, policy_model_name, limit_step):
    agent = DQN(env=env, policy_model_name=policy_model_name, animation_mode=False)

    env.reset()
    _, last_screen = env.get_screen()
    screen, current_screen = env.get_screen()
    state = current_screen - last_screen

    plt.ion()
    _, ax = plt.subplots()
    img = ax.imshow(screen)
    plt.pause(0.001)

    print('GAME START')

    N_TRIALS = 100
    succeed_cnt, fail_cnt = 0, 0
    for n_trial in range(N_TRIALS):
        step=0
        
        while True:
            step+=1
            if step > limit_step:
                fail_cnt += 1
                break
            action = agent.get_action(state)
            _, reward, done, _, _ = env.step(action)

            last_screen = current_screen
            screen, current_screen = env.get_screen()
            state = current_screen - last_screen

            img.set_data(screen)
            plt.pause(0.0001)
            print(f'[STEP:{step}] ACTION : {action}-{ACTIONS[action]}')

            time.sleep(1)
            
            if done:
                if reward > 0:
                    print('SUCCEEDED')
                else:
                    print('FAILED')
                break
        
        print(f'\rSUCCEED: {succeed_cnt:>2}, FAIL: {fail_cnt:>2}')
        
        env.reset()
        _, last_screen = env.get_screen()
        screen, current_screen = env.get_screen()
        state = current_screen - last_screen
        img.set_data(screen)

    plt.ioff()

if __name__ == "__main__":
    config = Config()
    env = Environment(config.env_config)
    agent = DQN(config, env, animation_mode=False)
    agent.learn()
    policy_model_name = input('policy weight file name: ').strip()
    # policy_model_name = 'policy_model_1000_20250428_16h47m.pt'
    policy_model_name = config.dqn_config.weight_path + '/' + policy_model_name
    # model_name.replace("\\", "/")
    evaluate_dqn(env, policy_model_name, 15)