import gym
import numpy as np
import matplotlib.pyplot as plt


font = {'family': 'Times New Roman', 'size': 12}


env = gym.make('LunarLander-v2')
data = [[] for _ in range(8)]
for i_episode in range(1000):
    env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        for i in range(len(observation)):
            data[i].append(observation[i])
        if done:
            break

names = ['x histogram', 'y histogram', 'x velocity histogram', 'y velocity histogram',
         'angle histogram', 'angular velocity histogram', 'left leg histogram', 'right leg histogram']

# plt.rc('font', **font)
for n in range(8):
    ary = np.array(data[n])
    plt.hist(ary, bins='auto')
    plt.title(names[n])
    plt.xlabel('value')
    plt.ylabel('frequency')
    filename = names[n].replace(' ', '_') + '.png'
    plt.savefig(filename)
    plt.show()
