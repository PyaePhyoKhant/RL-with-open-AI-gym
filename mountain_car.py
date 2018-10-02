import gym.spaces
import numpy as np
from helpers.q_learning import QLearningAgent

# important global parameters
MAX_POS = 0.6
MIN_POS = -1.2
MAX_VEL = 0.07
MIN_VEL = -0.07
LEARNING_EPISODES = 3000
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.3
BINS = 20
NUMPY_BINS = BINS + 1
ANIMATION = True
TIME_LIMIT = 200  # robot should reach goal after 200 time steps

pos_qtz = np.linspace(MIN_POS, MAX_POS, BINS)
vel_qtz = np.linspace(MIN_VEL, MAX_VEL, BINS)

env = gym.make('MountainCar-v0')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n), (NUMPY_BINS, NUMPY_BINS, env.action_space.n))


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (pos, vel) = obs
    # digitize return array
    pos = int(np.digitize(pos, pos_qtz))
    vel = int(np.digitize(vel, vel_qtz))
    return pos, vel


# Learning
for i_episode in range(LEARNING_EPISODES):
    observation = env.reset()
    total_reward = 0
    if i_episode % 500 == 0:
        print(str(i_episode) + '/' + str(LEARNING_EPISODES) + ' training episodes complete')
    for _ in range(TIME_LIMIT):
        # get action
        old_state = extract_state(observation)
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        next_state = extract_state(observation)

        # update learner
        learner.update(old_state, action, next_state, reward)

        if done:
            break

# Testing
learner.set_epsilon(0)  # turn off exploration
for _ in range(TESTING_EPISODES):
    observation = env.reset()
    total_reward = 0
    for t in range(TIME_LIMIT):
        if ANIMATION:
            env.render()

        # get action
        old_state = extract_state(observation)
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)

        if done:
            if t < TIME_LIMIT - 1:
                print('success')
            else:
                print('fail')
            break
