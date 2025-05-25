import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from maze import Maze
from agent import QLearningAgent
from memory import PrioritizedReplayBuffer
from rewards import get_reward


def animate_path(env, path):
    """Animate the path finding process"""
    plt.figure(figsize=(8, 8))
    grid = np.zeros((env.size, env.size))

    # Initialize the map
    for (x, y) in env.obstacles:
        grid[x, y] = 1  # Obstacles
    grid[env.start] = 2  # Start point
    grid[env.goal] = 3   # Goal

    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])

    for step, (x, y) in enumerate(path):
        # Clear current state
        if step > 0:
            grid[path[step - 1][0], path[step - 1][1]] = 0  # Clear previous position

        grid[x, y] = 4  # Mark current position

        plt.clf()
        plt.imshow(grid.T, cmap=cmap)

        # Add grid and labels
        plt.gca().set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        plt.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        plt.title(f"Step {step}: Position ({x}, {y})")

        # Pause for display
        plt.pause(0.5)  # Pause 0.5s per step

        # Stop if reached goal
        if (x, y) == env.goal:
            plt.pause(2)  # Pause 2s longer at goal
            break

    plt.show()

def train():
    env = Maze(size=10)
    agent = QLearningAgent(env.size,env.size)
    memory = PrioritizedReplayBuffer(10000)

    best_path = None
    best_steps = float('inf')

    for episode in range(500):
        state = env.reset()
        done = False
        current_path = [state]  # Record current path

        while not done:
            action = agent.get_action(state)
            next_state, done = env.step(action)
            reward = get_reward(state, next_state, done, env)
            memory.add(state, action, reward, next_state, done)
            current_path.append(next_state)

            if len(memory) >= 32:
                indices, batch, weights = memory.sample(32)
                td_errors = agent.learn(batch, weights)
                memory.update_priorities(indices, td_errors)

            state = next_state

            if len(current_path) > 100:  # Prevent infinite loop
                break

        # Update best path
        if done and len(current_path) < best_steps:
            best_steps = len(current_path)
            best_path = current_path
            print(f"New best path! Steps: {best_steps}, Episode: {episode}")

    # Visualization after training
    print("\nTraining completed! Animating best path...")
    animate_path(env, best_path)


if __name__ == "__main__":
    train()