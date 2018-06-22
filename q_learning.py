from collections import defaultdict
import random


class QLearningAgent:
    def __init__(self, env, alpha, gamma):
        """
        :param env: gym environment
        :param alpha: learning rate
        :param gamma: discount
        """
        self.env = env
        self.values = defaultdict(int)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount

    def update(self, old_state, action, next_state, reward):
        next_max_q = max([self.values[(next_state, next_action)] for next_action in range(self.env.action_space.n)])
        sample_value = reward + self.gamma * next_max_q
        self.values[(old_state, action)] = (1 - self.alpha) * self.values[(old_state, action)] + self.alpha * sample_value

    def get_action(self, state):
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
