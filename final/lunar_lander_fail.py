import gym.spaces
from helpers.q_learning import QLearningAgent
import numpy as np
import winsound

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
LEARNING_EPISODES = 1000
TESTING_EPISODES = 100
LEARNING_RATE = 0.3
DISCOUNT = 0.9
EXPLORATION = 0.4
REAL_BINS = 9
BINS = REAL_BINS + 1
NUMPY_BINS = REAL_BINS + 1
ANIMATION = True
# train and test with learned data
USE_EXTERNAL = False

x_qtz = np.linspace(MIN_X, MAX_X, BINS)
y_qtz = np.linspace(MIN_Y, MAX_Y, BINS)
x_vel_qtz = np.linspace(MIN_X_VEL, MAX_X_VEL, BINS)
y_vel_qtz = np.linspace(MIN_Y_VEL, MAX_Y_VEL, BINS)
un1_qtz = np.linspace(-MAX_UN, MAX_UN, BINS)
un2_qtz = np.linspace(-MAX_UN, MAX_UN, BINS)

env = gym.make('LunarLander-v2')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n), (NUMPY_BINS, NUMPY_BINS, NUMPY_BINS, NUMPY_BINS, NUMPY_BINS, NUMPY_BINS, 2, 2, env.action_space.n))


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    def qtz(val, lns):
        binn = int(np.digitize(val, lns))
        if binn == 0:
            return 1
        elif binn == BINS:
            return BINS - 1
        else:
            return binn

    (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = obs
    x = qtz(x, x_qtz)
    y = qtz(y, y_qtz)
    x_vel = qtz(x_vel, x_vel_qtz)
    y_vel = qtz(y_vel, y_vel_qtz)
    un1 = qtz(unknown1, un1_qtz)
    un2 = qtz(unknown2, un2_qtz)
    leg1 = int(leg1)
    leg2 = int(leg2)
    return x, y, x_vel, y_vel, un1, un2, leg1, leg2


# Learning
reward_summation = 0
last_100_reward = []
# +1 so that average reward for last 100 episodes is shown
for i_episode in range(LEARNING_EPISODES+1):
    if i_episode % 100 == 0 and i_episode != 0:
        print(str(i_episode) + '/' + str(LEARNING_EPISODES) + ' training episodes complete')
        print('current avg reward:', sum(last_100_reward) / (len(last_100_reward)+1))
        last_100_reward = []
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
        learner.update(old_state, action, next_state, reward)

        total_reward += reward
        if done:
            last_100_reward.append(total_reward)
            reward_summation += total_reward
            break

# Testing
learner.set_epsilon(0)
reward_summation = 0
for i_episode in range(TESTING_EPISODES):
    observation = env.reset()
    total_reward = 0
    for _ in range(1000):
        if ANIMATION:
            if i_episode > TESTING_EPISODES - 10:
                env.render()
        # get action
        old_state = extract_state(observation)
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            reward_summation += total_reward
            break
print('Average testing reward: ', reward_summation / TESTING_EPISODES)
winsound.Beep(1000, 250)