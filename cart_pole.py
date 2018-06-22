import gym
import gym.spaces
import time
from quantizer import Quantizer
from q_learning import QLearningAgent

dist_qtz = Quantizer(-2.5, 2.5, 1700)  # 1700 bins so that -0.14 to 0.14 have 100 bins
ang_qtz = Quantizer(-0.30, 0.30, 100)  # -12 to 12 degree is -0.20944 to 0.20944 in radians

(dist, v1, ang, v2) = (0, 0, 0, 0)

env = gym.make('CartPole-v0')
learner = QLearningAgent(env, 0.2, 0.8)

# Learning
reward_list = []
for i_episode in range(10000):
    observation = env.reset()
    (dist, v1, ang, v2) = observation
    dist = dist_qtz.round(dist)
    ang = ang_qtz.round(ang)
    total_reward = 0
    for t in range(1000):
        # get action
        old_state = (dist, ang)
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (dist, v1, ang, v2) = observation
        dist = dist_qtz.round(dist)
        ang = ang_qtz.round(ang)
        next_state = (dist, ang)

        # update learner
        learner.update(old_state, action, next_state, reward)

        total_reward += reward
        if done:
            reward_list.append(total_reward)
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Best learning reward: ', max(reward_list))

# Testing
reward_list = []
for i_episode in range(10):
    observation = env.reset()
    (dist, v1, ang, v2) = observation
    dist = dist_qtz.round(dist)
    ang = ang_qtz.round(ang)
    total_reward = 0
    for t in range(1000):
        env.render()

        # get action
        old_state = (dist, ang)
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (dist, v1, ang, v2) = observation
        dist = dist_qtz.round(dist)
        ang = ang_qtz.round(ang)
        next_state = (dist, ang)

        time.sleep(0.1)
        total_reward += reward
        if done:
            reward_list.append(total_reward)
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Best testing reward: ', max(reward_list))
