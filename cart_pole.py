import gym
import gym.spaces
import time
from quantizer import Quantizer
from q_learning import QLearningAgent


# important global parameters
MAX_DIST = 2.5
MAX_RAD = 0.3
LEARNING_EPISODES = 30000
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.8
EXPLORATION = 0.2

ang_qtz = Quantizer(-MAX_RAD, MAX_RAD, 1000)  # -12 to 12 degree is -0.20944 to 0.20944 in radians

(dist, v1, ang, v2) = (0, 0, 0, 0)

env = gym.make('CartPole-v0')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION)


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (dist, v1, ang, v2) = obs
    return ang

# Learning
reward_list = []
for i_episode in range(LEARNING_EPISODES):
    env.reset()
    total_reward = 0
    for _ in range(1000):
        # get action
        ang = ang_qtz.round(ang)
        old_state = extract_state((dist, v1, ang, v2))
        if i_episode > 5000:
            a = 1
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (dist, v1, ang, v2) = observation
        ang = ang_qtz.round(ang)
        next_state = extract_state((dist, v1, ang, v2))

        # update learner
        if i_episode > 5000:
            a = 1
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
        # env.render()

        # get action
        ang = ang_qtz.round(ang)
        old_state = extract_state((dist, v1, ang, v2))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (dist, v1, ang, v2) = observation
        ang = ang_qtz.round(ang)
        next_state = extract_state((dist, v1, ang, v2))

        # time.sleep(0.1)
        total_reward += reward
        if done:
            reward_list.append(total_reward)
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Average testing reward: ', sum(reward_list)/len(reward_list))

