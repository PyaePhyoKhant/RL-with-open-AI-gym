import gym.spaces
from helpers.q_learning import QLearningAgent
import numpy as np

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
# special number for legs
LEG1 = 4
LEG2 = 5
LEARNING_EPISODES = 100
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.1
TEMP_BINS = 9
BINS = TEMP_BINS + 1
NUMPY_BINS = TEMP_BINS + 1
ANIMATION = True
# train and test with learned data
USE_EXTERNAL = True

x_qtz = np.linspace(MIN_X, MAX_X, BINS)
y_qtz = np.linspace(MIN_Y, MAX_Y, BINS)
x_vel_qtz = np.linspace(MIN_X_VEL, MAX_X_VEL, BINS)
y_vel_qtz = np.linspace(MIN_Y_VEL, MAX_Y_VEL, BINS)
un1_qtz = np.linspace(-MAX_UN, MAX_UN, BINS)
un2_qtz = np.linspace(-MAX_UN, MAX_UN, BINS)

(x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = (0, 0, 0, 0, 0, 0, 0, 0)

env = gym.make('LunarLander-v2')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n), (NUMPY_BINS, NUMPY_BINS, NUMPY_BINS, NUMPY_BINS, 2, 2, env.action_space.n))

if USE_EXTERNAL:
    print('loaded')
    learner.values = np.load('fail.npy')


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
    return x_vel, y_vel, un1, un2, leg1, leg2


# Learning
reward_summation = 0
last_100_reward = []
# +1 so that average reward for last 100 episodes is shown
for i_episode in range(LEARNING_EPISODES+1):
    if i_episode % 100 == 0:
        print(str(i_episode) + '/' + str(LEARNING_EPISODES) + ' training episodes complete')
        print('current avg reward:', sum(last_100_reward) / (len(last_100_reward)+1))
        last_100_reward = []
    (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = env.reset()
    total_reward = 0
    for _ in range(1000):
        # testing show
        if i_episode == (LEARNING_EPISODES - 100):
            learner.set_epsilon(0)
        if i_episode > (LEARNING_EPISODES - 10):
            if ANIMATION:
                env.render()

        # get action
        old_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))
        action = learner.get_action(old_state, i_episode, LEARNING_EPISODES)

        # one step
        observation, reward, done, info = env.step(action)
        (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation
        next_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))

        total_reward += reward
        if done:
            # testing show
            if i_episode > (LEARNING_EPISODES - 10):
                if ANIMATION:
                    print(total_reward)

            last_100_reward.append(total_reward)
            reward_summation += total_reward
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Average learning reward: ', reward_summation / LEARNING_EPISODES)

# if USE_EXTERNAL:
#     print('saved')
#     np.save('fail', learner.values)
