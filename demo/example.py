import gym

env = gym.make('LunarLander-v2')
for i_episode in range(10):
    env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
