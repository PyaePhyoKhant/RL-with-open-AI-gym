import matplotlib.pyplot as plt
import gym.spaces
import numpy as np

# important global parameters
MAX_POS = 0.6
MIN_POS = -1.2
MAX_VEL = 0.07
MIN_VEL = -0.07
LEARNING_EPISODES = 2500
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.3
BINS = 20
ANIMATION = True
TIME_LIMIT = 200  # robot should reach goal after 200 time steps

pos_qtz = np.linspace(MIN_POS, MAX_POS, BINS)
vel_qtz = np.linspace(MIN_VEL, MAX_VEL, BINS)

(pos, vel) = (0, 0)

env = gym.make('MountainCar-v0')


data = []
data2 = []
def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (pos, vel) = obs
    data.append(pos)
    # digitize return array
    pos = int(np.digitize(pos, pos_qtz))
    vel = int(np.digitize(vel, vel_qtz))
    data2.append(pos)
    return pos, vel


# Learning
for i_episode in range(LEARNING_EPISODES):
    env.reset()
    for _ in range(TIME_LIMIT):
        # get action
        old_state = extract_state((pos, vel))
        action = env.action_space.sample()

        # one step
        observation, reward, done, info = env.step(action)

        if done:
            break

plt.hist(data, bins='auto')
plt.title("real histogram")
plt.show()

plt.hist(data2, bins='auto')
plt.title("error histogram")
plt.show()
