import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque


class Maze:
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # Initialize action_effects first
        self.action_effects = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

        # Then generate maze (will use action_effects)
        self.obstacles = set()
        self._generate_valid_maze()

        self.state = self.start
        self.fig, self.ax = None, None

    def _generate_valid_maze(self):
        """Generate a maze with guaranteed path"""
        all_positions = [(i, j) for i in range(self.size)
                         for j in range(self.size)
                         if (i, j) not in [self.start, self.goal]]
        np.random.shuffle(all_positions)

        self.obstacles = set()
        for pos in all_positions:
            temp_obstacles = self.obstacles | {pos}
            if self._path_exists(temp_obstacles):
                self.obstacles = temp_obstacles

            if len(self.obstacles) >= self.size ** 2 * 0.2:
                break

    def _path_exists(self, obstacles):
        """Check if path exists using BFS"""
        visited = set()
        queue = deque([self.start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal:
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
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.action_effects[action]
        new_x, new_y = x + dx, y + dy

        if (0 <= new_x < self.size and 0 <= new_y < self.size and
                (new_x, new_y) not in self.obstacles):
            self.state = (new_x, new_y)

        done = (self.state == self.goal)
        return self.state, done

    def render(self):
        """Visualize the maze"""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        grid = np.zeros((self.size, self.size))

        # Set obstacles
        for (x, y) in self.obstacles:
            grid[x, y] = 1

        grid[self.start] = 2  # Start point
        grid[self.goal] = 3   # Goal
        grid[self.state] = 4  # Current position

        cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])
        self.ax.clear()
        self.ax.imshow(grid.T, cmap=cmap)

        # Add grid lines
        self.ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        self.ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        self.ax.tick_params(which="minor", size=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        plt.title(f"Maze (Size: {self.size}x{self.size})")
        plt.pause(0.1)