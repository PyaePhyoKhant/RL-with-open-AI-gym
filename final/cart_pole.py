import time
import numpy as np
import gym.spaces
from helpers.q_learning import QLearningAgent

# important global parameters
MAX_DIST = 2.5
MAX_RAD = 0.21
MAX_CART_VEL = 2.0
MAX_TIP_VEL = 2.0
LEARNING_EPISODES = 1000
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.2
BINS = 20
NUMPY_BINS = BINS + 1
ANIMATION = False

ang_qtz = np.linspace(-MAX_RAD, MAX_RAD, BINS)  # -12 to 12 degree is -0.20944 to 0.20944 in radians
cart_qtz = np.linspace(-MAX_CART_VEL, MAX_CART_VEL, BINS)
tip_qtz = np.linspace(-MAX_TIP_VEL, MAX_TIP_VEL, BINS)

env = gym.make('CartPole-v0')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n), (NUMPY_BINS, NUMPY_BINS, env.action_space.n))


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (dist, cart_vel, ang, tip_vel) = obs
    ang = int(np.digitize(ang, ang_qtz))
    cart_vel = int(np.digitize(cart_vel, cart_qtz))
    tip_vel = int(np.digitize(tip_vel, tip_qtz))
    return ang, cart_vel


# Learning
reward_list = [0]  # 0 is to avoid error when LEARNING_EPISODES is zero
for _ in range(LEARNING_EPISODES):
    observation = env.reset()
    total_reward = 0
    for _ in range(1000):
        # get action
        old_state = extract_state(observation)
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        next_state = extract_state(observation)

        # update learner
        (dist, cart_vel, ang, tip_vel) = observation
        reward -= abs(ang) * 10  # this increase average score significantly
        learner.update(old_state, action, next_state, reward)

        total_reward += reward
        if done:
            reward_list.append(total_reward)
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Average learning reward: ', sum(reward_list) / len(reward_list))

# Testing
learner.set_epsilon(0)  # turn off exploration
reward_list = []
for _ in range(TESTING_EPISODES):
    observation = env.reset()
    total_reward = 0
    for _ in range(1000):
        if ANIMATION:
            env.render()
            time.sleep(0.1)

        # get action
        old_state = extract_state(observation)
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)

        # time.sleep(0.1)
        total_reward += reward
        if done:
            reward_list.append(total_reward)
            break
print('Average testing reward: ', sum(reward_list) / len(reward_list))
