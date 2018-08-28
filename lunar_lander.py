import time
import gym.spaces
from helpers.q_learning import QLearningAgent
from helpers.quantizer import Quantizer

# important global parameters
MAX_X = 0.2
MIN_X = -0.2
MAX_Y = 1
MIN_Y = 0
MAX_VEL = 1.5
LEARNING_EPISODES = 2000
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.2
BINS = 10
ANIMATION = True

x_qtz = Quantizer(-MIN_X, MAX_X, BINS)
x_vel_qtz = Quantizer(-MAX_VEL, MAX_VEL, BINS)
y_vel_qtz = Quantizer(-MAX_VEL, MAX_VEL, BINS)

(x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = (0, 0, 0, 0, 0, 0, 0, 0)

env = gym.make('LunarLander-v2')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n))


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = obs
    x = x_qtz.round(x)
    x_vel = x_vel_qtz.round(x_vel)
    y_vel = y_vel_qtz.round(y_vel)
    return x, x_vel, y_vel


# Learning
reward_list = [0]   # 0 is to avoid error when LEARNING_EPISODES is zero
for _ in range(LEARNING_EPISODES):
    env.reset()
    total_reward = 0
    for _ in range(1000):
        # get action
        old_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation
        next_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))

        # update learner
        learner.update(old_state, action, next_state, reward)

        total_reward += reward
        if done:
            reward_list.append(total_reward)
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Average learning reward: ', sum(reward_list)/len(reward_list))

# Testing
learner.set_epsilon(0)  # turn off exploration
reward_list = []
for _ in range(TESTING_EPISODES):
    env.reset()
    total_reward = 0
    for _ in range(1000):
        if ANIMATION:
            env.render()
            # time.sleep(0.1)

        # get action
        old_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation

        # time.sleep(0.1)
        total_reward += reward
        if done:
            reward_list.append(total_reward)
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Average testing reward: ', sum(reward_list)/len(reward_list), ' (', TESTING_EPISODES, ' trials)')

