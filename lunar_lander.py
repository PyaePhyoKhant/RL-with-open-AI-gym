import time
import gym.spaces
from helpers.q_learning import QLearningAgent
from helpers.quantizer import Quantizer

# important global parameters
MAX_X = 0.3
MIN_X = -0.3
MAX_Y = 1
MIN_Y = 0
MAX_X_VEL = 1
MIN_X_VEL = -1
MAX_Y_VEL = 0.5
MIN_Y_VEL = -1.5
MAX_UN = 1
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.95
EXPLORATION = 0.2
BINS = 5
ANIMATION = False

x_qtz = Quantizer(MIN_X, MAX_X, BINS)
y_qtz = Quantizer(MIN_Y, MAX_Y, BINS)
x_vel_qtz = Quantizer(MIN_X_VEL, MAX_X_VEL, BINS)
y_vel_qtz = Quantizer(MIN_Y_VEL, MAX_Y_VEL, BINS)
un1_qtz = Quantizer(-MAX_UN, MAX_UN, BINS)
un2_qtz = Quantizer(-MAX_UN, MAX_UN, BINS)

(x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = (0, 0, 0, 0, 0, 0, 0, 0)

env = gym.make('LunarLander-v2')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n), (BINS, BINS, BINS, BINS, BINS, BINS, env.action_space.n))


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = obs
    x = x_qtz.value_to_index(x)
    y = y_qtz.value_to_index(y)
    x_vel = x_vel_qtz.value_to_index(x_vel)
    y_vel = y_vel_qtz.value_to_index(y_vel)
    un1 = un1_qtz.value_to_index(unknown1)
    un2 = un2_qtz.value_to_index(unknown2)
    return x, y, x_vel, y_vel, un1, un2


# Learning
reward_summation = 0  # 0 is to avoid error when LEARNING_EPISODES is zero
st = time.time()
for i_episode in range(LEARNING_EPISODES):
    if i_episode % 100 == 0:
        print(str(i_episode) + '/' + str(LEARNING_EPISODES) + ' training episodes complete')
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
            reward_summation += total_reward
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Average learning reward: ', reward_summation / LEARNING_EPISODES)
print('time: ', time.time() - st)

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
print('Average testing reward:', sum(reward_list) / len(reward_list), '(', TESTING_EPISODES, 'trials)')
