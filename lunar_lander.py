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
# special number for legs
LEG1 = 4
LEG2 = 5
LEARNING_EPISODES = 3000
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.3
BINS = 10
ANIMATION = True

x_qtz = Quantizer(MIN_X, MAX_X, BINS)
y_qtz = Quantizer(MIN_Y, MAX_Y, BINS)
x_vel_qtz = Quantizer(MIN_X_VEL, MAX_X_VEL, BINS)
y_vel_qtz = Quantizer(MIN_Y_VEL, MAX_Y_VEL, BINS)
un1_qtz = Quantizer(-MAX_UN, MAX_UN, BINS)
un2_qtz = Quantizer(-MAX_UN, MAX_UN, BINS)

(x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = (0, 0, 0, 0, 0, 0, 0, 0)

env = gym.make('LunarLander-v2')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n))


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = obs
    # x = x_qtz.round(x)
    # y = y_qtz.round(y)
    x_vel = x_vel_qtz.round(x_vel)
    y_vel = y_vel_qtz.round(y_vel)
    un1 = un1_qtz.round(unknown1)
    un2 = un2_qtz.round(unknown2)
    # if leg1 == 1:
    #     x = LEG1
    # if leg2 == 1:
    #     y = LEG2
    return leg1, leg2, x_vel, y_vel, un1, un2


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
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation
        next_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))

        # update learner
        learner.update(old_state, action, next_state, reward)

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

# # Testing
# learner.set_epsilon(0)  # turn off exploration
# reward_list = []
# for ep in range(TESTING_EPISODES):
#     (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = env.reset()
#     total_reward = 0
#     for _ in range(1000):
#         if ANIMATION:
#             if ep < 10:
#                 env.render()
#
#         # get action
#         old_state = extract_state((x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2))
#         action = learner.get_action(old_state)
#
#         # one step
#         observation, reward, done, info = env.step(action)
#         (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation
#
#         total_reward += reward
#         if done:
#             print(total_reward)
#             reward_list.append(total_reward)
#             # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
#             break
# print('Average testing reward:', sum(reward_list) / len(reward_list), '(', TESTING_EPISODES, 'trials )')
