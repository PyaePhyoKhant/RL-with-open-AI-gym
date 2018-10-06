import gym.spaces
from helpers.q_learning import QLearningAgent
import numpy as np
import pickle
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
DISCOUNT = 0.9
EXPLORATION = 0.0
BINS = 8

x_qtz = np.linspace(MIN_X, MAX_X, BINS)
y_qtz = np.linspace(MIN_Y, MAX_Y, BINS)
x_vel_qtz = Quantizer(MIN_X_VEL, MAX_X_VEL, BINS)
y_vel_qtz = Quantizer(MIN_Y_VEL, MAX_Y_VEL, BINS)
un1_qtz = Quantizer(-MAX_UN, MAX_UN, BINS)
un2_qtz = Quantizer(-MAX_UN, MAX_UN, BINS)

(x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = (0, 0, 0, 0, 0, 0, 0, 0)

env = gym.make('LunarLander-v2')
env.seed(2)
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n))
with open('lunar_lander_knowledge_bins_8.txt', 'rb') as f:
    learner.values = pickle.load(f)


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = obs
    # x = qtz(x, x_qtz)
    # y = qtz(y, y_qtz)
    x_vel = x_vel_qtz.round(x_vel)
    y_vel = y_vel_qtz.round(y_vel)
    un1 = un1_qtz.round(unknown1)
    un2 = un2_qtz.round(unknown2)
    return leg1, leg2, x_vel, y_vel, un1, un2


# Learning
reward_summation = 0
# +1 so that average reward for last 100 episodes is shown
for i_episode in range(TESTING_EPISODES):
    (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = env.reset()
    total_reward = 0
    for _ in range(1000):
        env.render()

        # get action
        old_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation
        next_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))

        total_reward += reward
        if done:
            reward_summation += total_reward
            break
print('Average testing reward: ', reward_summation / TESTING_EPISODES)
