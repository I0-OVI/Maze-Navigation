import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, maze_size_x, maze_size_y, action_size=4):
        # Expand the Q-table to include key status
        self.q_table = np.zeros((maze_size_x, maze_size_y, 2, action_size))  # New third dimension: 0=no key, 1=has key
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Increase the minimum exploration rate
        self.epsilon_decay = 0.995  # Slow down the decay
        self.state_action_counts = defaultdict(int)  # Track state-action visit counts

    def get_action(self, state, has_key=False):
        """
        Parameters:
            state: Current coordinates (x, y)
            has_key: Whether holding the key
        """
        x, y = map(int, state)
        key_state = int(has_key)

        # Exploration based on counts (UCB heuristic)
        if np.random.random() < self.epsilon:
            # Prefer actions with fewer visits
            action_counts = [self.state_action_counts[(x, y, key_state, a)] for a in range(4)]
            return np.argmin(action_counts)

        return np.argmax(self.q_table[x, y, key_state])

    def learn(self, batch, env):
        """
        Parameters:
            batch: [(state, action, reward, next_state, done), ...]
            env: Maze environment object, used to query key status
        """
        states, actions, rewards, next_states, dones = zip(*batch)
        td_errors = []

        for i in range(len(batch)):
            x, y = map(int, states[i])
            next_x, next_y = map(int, next_states[i])

            # Directly get key status from the environment
            current_has_key = env.has_key or ((x, y) == env.key_pos and not env.has_key)
            next_has_key = env.has_key or ((next_x, next_y) == env.key_pos and not env.has_key)

            # Calculate TD target
            if dones[i]:
                td_target = rewards[i]
            else:
                td_target = rewards[i] + self.gamma * np.max(
                    self.q_table[next_x, next_y, int(next_has_key)]
                )

            # Update Q value
            td_error = td_target - self.q_table[x, y, int(current_has_key), actions[i]]
            self.q_table[x, y, int(current_has_key), actions[i]] += self.lr * td_error
            td_errors.append(abs(td_error))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return td_errors