"""
this is not working with numpy for now
"""

import time
import gym.spaces
from helpers.q_learning import QLearningAgent
from helpers.quantizer import Quantizer
import pickle

# important global parameters
MAX_COS_THETA = 1.0
MAX_SIN_THETA = 1.0
MAX_THETA_DOT = 8.0
LEARNING_EPISODES = 0
TESTING_EPISODES = 10
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.3
BINS = 10
ANIMATION = True
# set True when skipping learning phase and load knowledge from external with 5000 LEARNING_EPISODES
USE_EXTERNAL = True

cos_qtz = Quantizer(-MAX_COS_THETA, MAX_COS_THETA, BINS)
sin_qtz = Quantizer(-MAX_SIN_THETA, MAX_SIN_THETA, BINS)
theta_qtz = Quantizer(-MAX_THETA_DOT, MAX_THETA_DOT, BINS)

(c, s, d) = (0, 0, 0)

env = gym.make('Pendulum-v0')
action_qtz = Quantizer(-2.0, 2.0, 10)
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, action_qtz.as_list())


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (c, s, d) = obs
    c = cos_qtz.round(c)
    s = sin_qtz.round(s)
    d = theta_qtz.round(d)
    return c, s, d

# Learning
for i_episode in range(LEARNING_EPISODES):
    env.reset()
    if i_episode % 500 == 0:
        print(str(i_episode) + '/' + str(LEARNING_EPISODES) + ' training episodes complete')
    for _ in range(1000):
        # get action
        old_state = extract_state((c, s, d))
        action = learner.get_action(old_state)

        # one step
        # typecast action to list for continuous space
        # https://github.com/openai/gym/issues/602
        observation, reward, done, info = env.step([action])
        (c, s, d) = observation
        next_state = extract_state((c, s, d))

        # update learner
        learner.update(old_state, action, next_state, reward)

        if done:
            break

# if USE_EXTERNAL==True, use knowledge with 5000 LEARNING_EPISODES
if USE_EXTERNAL:
    print("Skipping learning and load knowledge from external")
    with open('pendulum_knowledge.txt', 'rb') as f:
        learner.values = pickle.load(f)

# Testing
learner.set_epsilon(0)  # turn off exploration
for _ in range(TESTING_EPISODES):
    env.reset()
    for _ in range(100000):
        if ANIMATION:
            env.render()

        # get action
        old_state = extract_state((c, s, d))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step([action])
        (c, s, d) = observation

        if done:
            time.sleep(2)
            break
