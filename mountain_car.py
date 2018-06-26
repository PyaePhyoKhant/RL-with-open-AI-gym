import time

import gym.spaces

from helpers.q_learning import QLearningAgent
from helpers.quantizer import Quantizer

# important global parameters
MAX_POS = 0.6
MIN_POS = -1.2
MAX_VEL = 0.07
MIN_VEL = -0.07
LEARNING_EPISODES = 2500
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.3
BINS = 20
ANIMATION = True
TIME_LIMIT = 200    # robot should reach goal after 200 time steps

pos_qtz = Quantizer(MIN_POS, MAX_POS, BINS)
vel_qtz = Quantizer(MIN_VEL, MAX_VEL, BINS)

(pos, vel) = (0, 0)

env = gym.make('MountainCar-v0')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n))


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (pos, vel) = obs
    pos = pos_qtz.round(pos)
    vel = vel_qtz.round(vel)
    return pos, vel

# Learning
for i_episode in range(LEARNING_EPISODES):
    env.reset()
    total_reward = 0
    if i_episode % 500 == 0:
        print(str(i_episode) + '/' + str(LEARNING_EPISODES) + ' training episodes complete')
    for _ in range(TIME_LIMIT):
        # get action
        old_state = extract_state((pos, vel))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (pos, vel) = observation
        next_state = extract_state((pos, vel))

        # update learner
        learner.update(old_state, action, next_state, reward)

        if done:
            break

# Testing
learner.set_epsilon(0)  # turn off exploration
for _ in range(TESTING_EPISODES):
    env.reset()
    total_reward = 0
    for t in range(TIME_LIMIT):
        if ANIMATION:
            env.render()

        # get action
        old_state = extract_state((pos, vel))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (pos, vel) = observation

        if done:
            if t < TIME_LIMIT-1:
                print('success')
            else:
                print('fail')
            break
