from collections import defaultdict
import random
import numpy as np


class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon, actions, table_size=None):
        """
        :param env: gym environment
        :param alpha: learning rate
        :param gamma: discount
        :param epsilon: exploration
        :param actions: list of available actions (e.g. [0,1])
        """
        self.env = env
        if table_size is not None:
            self.values = np.zeros(table_size)
        else:
            self.values = defaultdict(int)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # exploration
        self.actions = actions  # list of available actions
        self.use_numpy = table_size is not None

    def update(self, old_state, action, next_state, reward):
        if self.use_numpy:
            next_max_q = self.values[next_state].max()
        else:
            next_max_q = max([self.values[(*next_state, next_action)] for next_action in self.actions])
        sample_value = reward + self.gamma * next_max_q
        self.values[(*old_state, action)] = (1 - self.alpha) * self.values[(*old_state, action)] + self.alpha * sample_value

    def get_action(self, state, current_ep=None, total_ep=None):
        # choose random action (exploration)
        if current_ep is not None and total_ep is not None:
            threshold_for_random = (total_ep - current_ep) / total_ep
        else:
            threshold_for_random = self.epsilon

        if random.random() < threshold_for_random:
            return random.choice(self.actions)
        # choose best action (exploitation)
        else:
            if self.use_numpy:
                # https://stackoverflow.com/a/42071648/8211573
                b = self.values[state]
                return np.random.choice(np.flatnonzero(b == b.max()))
            else:
                max_actions = []
                max_value = float('-inf')
                for action in self.actions:
                    q = self.values[(*state, action)]
                    if q == max_value:
                        max_actions.append(action)
                    elif q > max_value:
                        max_actions = [action]
                        max_value = q
                return random.choice(max_actions)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
