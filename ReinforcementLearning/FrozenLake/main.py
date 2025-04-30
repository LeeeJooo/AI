import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment
from dqn import DQN
from config import Config
import time

ACTIONS = {0: 'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}

def evaluate_dqn(env, config, policy_model_name, limit_step):
    agent = DQN(env=env, config=config, policy_model_name=policy_model_name, animation_mode=False)

    env.reset()
    _screen = env.get_screen()

    plt.ion()
    _, ax = plt.subplots()
    img = ax.imshow(_screen)
    plt.pause(0.001)

    print('GAME START')

    N_TRIALS = 100
    succeed_cnt, fail_cnt = 0, 0
    for n_trial in range(N_TRIALS):
        observation = agent.preprocessing(0, None, None, 0, None, 0., False)
        step=0
        
        while True:
            step+=1
            if step > limit_step:
                fail_cnt += 1
                break
            action = agent.get_action(observation.cur_state)
            observation = agent.step(observation, action)

            _screen = env.get_screen()

            img.set_data(_screen)
            plt.pause(0.0001)
            print(f'[STEP:{step}] ACTION : {action}-{ACTIONS[action]}')

            time.sleep(1)
            
            if observation.done:
                if observation.reward > 0:
                    print('SUCCEEDED')
                    succeed_cnt += 1
                else:
                    print('FAILED')
                    fail_cnt += 1
                break
        
        print(f'\rSUCCEED: {succeed_cnt:>2}, FAIL: {fail_cnt:>2}')
        
        env.reset()
        _screen = env.get_screen()
        img.set_data(_screen)

    plt.ioff()

if __name__ == "__main__":
    config = Config()
    env = Environment(config.env_config)
    '''
    성공 모델 : 4x4, not slippery
    policy_model_10000_20250429_20h43m.pt : 8529 / 10000
    policy_model_10000_20250429_20h49m.pt : 8452 / 10000
    policy_model_10000_20250429_20h51m.pt : 8461 / 10000
    policy_model_10000_20250429_20h54m.pt : 8550 / 10000

    성공 모델 : 4x4, slippery
    491 / 10000
    10h54m : 460 / 10000 (pos-1)
    10h55m : 357 / 10000 (pos-1)
    10h57m : 385 / 10000 (pos-3)
    10h58m : 372 / 10000 (pos-5)
    11h00m : 313 / 10000 (pos-3, neg--1)
    '''
    policy_model_name = 'slippery_policy_model_10000_20250430_10h55m'
    policy_model_name = config.dqn_config.weight_path + '/' + policy_model_name + '.pt'
    target_model_name = 'slippery_target_model_10000_20250430_10h55m'
    target_model_name = config.dqn_config.weight_path + '/' + target_model_name + '.pt'
    agent = DQN(config, env, animation_mode=False, policy_model_name=policy_model_name, target_model_name=target_model_name)
    agent.learn()
    policy_model_name = input('policy weight file name: ').strip()
    policy_model_name = config.dqn_config.weight_path + '/' + policy_model_name + '.pt'
    evaluate_dqn(env, config, policy_model_name, 15)