import numpy as np
from collections import deque


class Maze:
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # Initialize action effects first
        self.action_effects = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

        # Then initialize other attributes
        self.key_pos = None
        self.has_key = False
        self.obstacles = set()
        self._generate_valid_maze()

    def _generate_valid_maze(self):
        """Generate a valid maze with uniformly distributed obstacles"""
        # Initialize empty maze
        grid = np.zeros((self.size, self.size), dtype=bool)

        # Ensure key position is at least 3 cells away from goal
        all_positions = [(i, j) for i in range(self.size)
                         for j in range(self.size)
                         if (i, j) not in [self.start, self.goal]]

        valid_key_pos = [pos for pos in all_positions
                         if abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1]) >= 3]
        self.key_pos = valid_key_pos[np.random.choice(len(valid_key_pos))]

        # Obstacle generation parameters
        target_obstacles = int(self.size ** 2 * 0.2)  # 20% obstacles
        max_attempts = 1000
        attempts = 0

        while len(self.obstacles) < target_obstacles and attempts < max_attempts:
            attempts += 1

            # Prefer generating obstacles on the right side (balanced distribution)
            if np.random.random() < 0.7 and self.size > 5:  # 70% probability on right side
                x = np.random.randint(self.size // 2, self.size)
                y = np.random.randint(self.size)
            else:
                x, y = np.random.randint(self.size), np.random.randint(self.size)

            pos = (x, y)

            # Skip critical positions
            if pos in [self.start, self.goal, self.key_pos]:
                continue

            temp_obs = self.obstacles | {pos}

            # Check path validity
            if self._path_exists(self.start, self.key_pos, temp_obs) and \
                    self._path_exists(self.key_pos, self.goal, temp_obs):
                self.obstacles = temp_obs

        # Fallback if not enough obstacles generated
        if len(self.obstacles) < target_obstacles * 0.8:
            print("Warning: Using fallback obstacle generation")
            self._fallback_obstacle_generation()

    def _fallback_obstacle_generation(self):
        """Fallback obstacle generation method"""
        # Generate random obstacles that don't block paths
        free_positions = [(i, j) for i in range(self.size)
                          for j in range(self.size)
                          if (i, j) not in [self.start, self.goal, self.key_pos] + list(self.obstacles)]

        need = int(self.size ** 2 * 0.2) - len(self.obstacles)
        self.obstacles.update(set(free_positions[:need]))

    def _path_exists(self, start, end, obstacles):
        """BFS to check if path exists"""
        visited = set()
        queue = deque([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == end:
                return True

            if (x, y) in visited or (x, y) in obstacles:
                continue

            visited.add((x, y))

            # Use pre-initialized action_effects
            for dx, dy in self.action_effects:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    queue.append((nx, ny))
        return False

    def reset(self):
        self.has_key = False
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.action_effects[action]
        new_x, new_y = x + dx, y + dy

        # Movement validity check
        if (0 <= new_x < self.size and 0 <= new_y < self.size and
                (new_x, new_y) not in self.obstacles):
            self.state = (new_x, new_y)

            # Check if key is obtained
            if self.state == self.key_pos:
                self.has_key = True

        # Goal check (must have key to pass)
        done = (self.state == self.goal) and self.has_key
        return self.state, done

    def bfs_distance(self, start, end, include_key=False):
        """Calculate shortest path distance between two points (considering obstacles)"""
        if start == end:
            return 0

        visited = set()
        queue = deque([(start, 0)])  # (position, distance)

        while queue:
            (x, y), dist = queue.popleft()

            # Check if reached destination
            if (x, y) == end:
                return dist

            # Skip visited and obstacles
            if (x, y) in visited or (x, y) in self.obstacles:
                continue

            visited.add((x, y))

            # Check key requirement
            if include_key and (x, y) == self.key_pos:
                continue  # If must have key but haven't obtained it

            # Explore four directions
            for dx, dy in self.action_effects:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    queue.append(((nx, ny), dist + 1))

        return float('inf')  # Unreachable

    def get_key_state(self, pos):
        """Get key state for specified position"""
        x, y = pos
        # Either already has key, or is at key position and hasn't obtained it yet
        return self.has_key or ((x, y) == self.key_pos and not self.has_key)