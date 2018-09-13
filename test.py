import gym
import matplotlib.pyplot as plt
import numpy as np

LEARNING_EPISODES = 1000

v = []
env = gym.make('LunarLander-v2')
for i_episode in range(LEARNING_EPISODES):
    if i_episode % 100 == 0:
        print(str(i_episode) + '/' + str(LEARNING_EPISODES) + ' training episodes complete')
    env.reset()
    for t in range(100):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation
        v.append(unknown2)
        if done:
            # print("Episode finished after {} timesteps".format(t + 1))
            break

data = np.array(v)
plt.hist(data, bins='auto')
plt.title("unknown2 histogram")
plt.show()


def filter(lst, low, high):
    new_list = []
    for n in lst:
        if low < n < high:
            new_list.append(n)
    return new_list


dummy = 'dummy'
