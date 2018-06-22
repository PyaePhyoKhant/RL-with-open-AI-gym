from collections import defaultdict
import random


class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        """
        :param env: gym environment
        :param alpha: learning rate
        :param gamma: discount
        :param epsilon: exploration
        """
        self.env = env
        self.values = defaultdict(int)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # exploration

    def update(self, old_state, action, next_state, reward):
        next_max_q = max([self.values[(next_state, next_action)] for next_action in range(self.env.action_space.n)])
        sample_value = reward + self.gamma * next_max_q
        self.values[(old_state, action)] = (1 - self.alpha) * self.values[(old_state, action)] + self.alpha * sample_value

    def get_action(self, state):
        # choose random action (exploration)
        if random.random() < self.epsilon:
            return random.choice(range(self.env.action_space.n))
        # choose best action (exploitation)
        else:
            max_actions = []
            max_value = float('-inf')
            for action in range(self.env.action_space.n):
                q = self.values[(state, action)]
                if q == max_value:
                    max_actions.append(action)
                elif q > max_value:
                    max_actions = [action]
                    max_value = q
            return random.choice(max_actions)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
