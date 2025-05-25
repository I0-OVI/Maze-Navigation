import numpy as np
from SumTree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(capacity)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0
        self.max_priority = 1.0

        # New addition: Attributes related to path memory
        self.visited_states = {}  # Record the number of times a state is visited {state: count}
        self.visit_decay = 0.99  # Decay coefficient for visit count

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done):


        # Convert the state to a hashable tuple
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Calculate the repeat visit penalty (new addition)
        visit_count = self.visited_states.get(state_key, 0)
        if visit_count > 0:
            reward -= 0.3 * visit_count   # The penalty coefficient can be adjusted as needed

        # Update the visit count (new addition)
        self.visited_states[state_key] = visit_count + 1
        self.visited_states[next_state_key] = self.visited_states.get(next_state_key, 0) + 1

        # Decay all visit counts (to prevent infinite growth)
        for k in self.visited_states:
            self.visited_states[k] *= self.visit_decay

        # Store the experience
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, data)
        self.data[self.write_pos] = data
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = []
        samples = []
        weights = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            prob = priority / self.tree.total()
            weight = (1.0 / prob) ** self.beta
            indices.append(idx)
            samples.append(data)
            weights.append(weight)

        weights = np.array(weights) / np.max(weights)
        return indices, samples, weights

    def update_priorities(self, indices, priorities):
        try:
            iter(priorities)
        except TypeError:
            priorities = [priorities] * len(indices)

        for idx, priority in zip(indices, priorities):
            priority = max(float(priority), 1e-6)
            self.tree.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)

    def clear_visited(self):
        self.visited_states = {}