import gym
import gym.spaces
import time
from quantizer import Quantizer
from q_learning import QLearningAgent


# important global parameters
MAX_DIST = 2.5
MAX_RAD = 0.21
MAX_CART_VEL = 2.0
MAX_TIP_VEL = 2.0
LEARNING_EPISODES = 10000
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.5
BINS = 100
ANIMATION = False

ang_qtz = Quantizer(-MAX_RAD, MAX_RAD, BINS)  # -12 to 12 degree is -0.20944 to 0.20944 in radians
cart_qtz = Quantizer(-MAX_CART_VEL, MAX_CART_VEL, BINS)
tip_qtz = Quantizer(-MAX_TIP_VEL, MAX_TIP_VEL, BINS)

(dist, cart_vel, ang, tip_vel) = (0, 0, 0, 0)

env = gym.make('CartPole-v0')
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION)


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (dist, cart_vel, ang, tip_vel) = obs
    ang = ang_qtz.round(ang)
    cart_vel = cart_qtz.round(cart_vel)
    tip_vel = tip_qtz.round(tip_vel)
    return ang, cart_vel

# Learning
reward_list = []
for _ in range(LEARNING_EPISODES):
    env.reset()
    total_reward = 0
    for _ in range(1000):
        # get action
        old_state = extract_state((dist, cart_vel, ang, tip_vel))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (dist, cart_vel, ang, tip_vel) = observation
        next_state = extract_state((dist, cart_vel, ang, tip_vel))

        # update learner
        reward -= abs(ang) * 10  # this increase average score significantly
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
        if ANIMATION:
            env.render()
            time.sleep(0.1)

        # get action
        old_state = extract_state((dist, cart_vel, ang, tip_vel))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step(action)
        (dist, cart_vel, ang, tip_vel) = observation

        # time.sleep(0.1)
        total_reward += reward
        if done:
            reward_list.append(total_reward)
            # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, total_reward))
            break
print('Average testing reward: ', sum(reward_list)/len(reward_list))

