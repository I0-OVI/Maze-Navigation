import numpy as np


class QLearningAgent:
    def __init__(self, maze_size_x, maze_size_y, action_size=4):
        """Initialize Q-learning agent with Q-table and learning parameters.

        Args:
            maze_size_x (int): Width of the maze
            maze_size_y (int): Height of the maze
            action_size (int): Number of possible actions (default: 4)
        """
        self.q_table = np.zeros((maze_size_x, maze_size_y, action_size))
        self.lr = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Initial high exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate

    def get_action(self, state, epsilon=0.1):
        """Select an action using ε-greedy policy.

        Args:
            state: Current state coordinates
            epsilon: Exploration probability (default: 0.1)

        Returns:
            int: Selected action index (0-3)
        """
        if np.random.random() < epsilon:
            return np.random.randint(4)  # Random action within fixed range

        x, y = map(int, state)  # Ensure coordinates are integers
        return np.argmax(self.q_table[x, y])  # Safe indexing for Q-table

    def learn(self, batch, weights):
        """Update Q-values using experience replay with prioritized sampling.

        Args:
            batch: List of experiences (state, action, reward, next_state, done)
            weights: Importance sampling weights

        Returns:
            list: TD errors for each experience in the batch
        """
        states, actions, rewards, next_states, dones = zip(*batch)
        td_errors = []

        for i in range(len(batch)):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]

            # Calculate TD target and error
            td_target = rewards[i] + self.gamma * np.max(self.q_table[next_state])
            td_error = td_target - self.q_table[state][action]

            # Update Q-value with weighted learning
            self.q_table[state][action] += self.lr * td_error * weights[i]
            td_errors.append(abs(td_error))

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return td_errors  # Return list of TD errors for each experience