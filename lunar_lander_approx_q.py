import gym.spaces
from helpers.approx_q import ApproximateQAgent


LEARNING_EPISODES = 3000
TESTING_EPISODES = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EXPLORATION = 0.9
ANIMATION = True
# train and test with learned data
USE_EXTERNAL = False


(x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = (0, 0, 0, 0, 0, 0, 0, 0)

env = gym.make('LunarLander-v2')
learner = ApproximateQAgent(env, LEARNING_RATE, DISCOUNT, EXPLORATION, range(env.action_space.n))


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
        if i_episode > 100:
            a = 1
        # testing show
        if i_episode == (LEARNING_EPISODES - 100):
            learner.set_epsilon(0)
        if i_episode > (LEARNING_EPISODES - 10):
            if ANIMATION:
                env.render()

        # get action
        old_state = (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2)
        action = learner.get_action(old_state, i_episode, LEARNING_EPISODES)

        # one step
        observation, reward, done, info = env.step(action)
        (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2) = observation
        next_state = (x, y, x_vel, y_vel, unknown1, unknown2, leg1, leg2)

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
            break
print('Average learning reward: ', reward_summation / LEARNING_EPISODES)
