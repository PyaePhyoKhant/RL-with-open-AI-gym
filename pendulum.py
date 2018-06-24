import gym.spaces
import time
from quantizer import Quantizer
from q_learning import QLearningAgent


# important global parameters
MAX_COS_THETA = 1.0
MAX_SIN_THETA = 1.0
MAX_THETA_DOT = 8.0
LEARNING_EPISODES = 4000
TESTING_EPISODES = 10
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.2
BINS = 10
ANIMATION = True

cos_qtz = Quantizer(-MAX_COS_THETA, MAX_COS_THETA, BINS)
sin_qtz = Quantizer(-MAX_SIN_THETA, MAX_SIN_THETA, BINS)
theta_qtz = Quantizer(-MAX_THETA_DOT, MAX_THETA_DOT, BINS)

(c, s, d) = (0, 0, 0)

env = gym.make('Pendulum-v0')
action_qtz = Quantizer(-2.0, 2.0, 10)
learner = QLearningAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, action_qtz.as_list())


def extract_state(obs):
    """
    extract state via this function so that it is DRY
    :param obs: gym observation
    """
    (c, s, d) = obs
    c = cos_qtz.round(c)
    s = sin_qtz.round(s)
    d = theta_qtz.round(d)
    return c, s, d

# Learning
for iep in range(LEARNING_EPISODES):
    env.reset()
    if iep % 500 == 0:
        print(iep)
    for _ in range(1000):
        # get action
        old_state = extract_state((c, s, d))
        action = learner.get_action(old_state)

        # one step
        # typecast action to list for continuous space
        # https://github.com/openai/gym/issues/602
        observation, reward, done, info = env.step([action])
        (c, s, d) = observation
        next_state = extract_state((c, s, d))

        # update learner
        learner.update(old_state, action, next_state, reward)

        if done:
            break

# Testing
learner.set_epsilon(0)  # turn off exploration
for _ in range(TESTING_EPISODES):
    env.reset()
    for _ in range(50000):
        if ANIMATION:
            env.render()

        # get action
        old_state = extract_state((c, s, d))
        action = learner.get_action(old_state)

        # one step
        observation, reward, done, info = env.step([action])
        (c, s, d) = observation

        if done:
            time.sleep(2)
            break
