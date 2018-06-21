import gym
import gym.spaces
import time
from quantizer import Quantizer

dist_qtz = Quantizer(-2.4, 2.4, 1700)  # 1700 bins so that -0.14 to 0.14 have 100 bins
ang_qtz = Quantizer(-0.20944, 0.20944, 100)  # -12 to 12 degree is -0.20944 to 0.20944 in radians

(dist, v1, ang, v2) = (0, 0, 0, 0)

env = gym.make('CartPole-v0')
for i_episode in range(1):
    observation = env.reset()
    total_reward = 0
    for t in range(1000):
        env.render()
        print(observation)
        if ang > 0:
            observation, reward, done, info = env.step(1)
        else:
            observation, reward, done, info = env.step(0)
        (dist, v1, ang, v2) = observation
        time.sleep(0.1)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
