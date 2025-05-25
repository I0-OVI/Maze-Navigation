import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from maze import Maze
from agent import QLearningAgent
from memory import PrioritizedReplayBuffer
from rewards import get_reward
import time


def animate_path(env, path):
    """Animate the path finding process step by step"""
    plt.figure(figsize=(8, 8))
    grid = np.zeros((env.size, env.size))

    # Initialize static elements
    for (x, y) in env.obstacles:
        grid[x, y] = 1  # Obstacles (black)
    grid[env.start] = 2  # Start point (green)
    grid[env.goal] = 3  # Goal (red)
    grid[env.key_pos] = 4  # Key (gold)

    # Create colormap
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'gold', 'blue'])
    print(path)
    for step in range(len(path)):
        plt.clf()

        # Copy base grid
        current_grid = grid.copy()

        # Mark current path point (blue)
        x, y = path[step]
        current_grid[x, y] = 5

        # Display
        plt.imshow(current_grid.T, cmap=cmap)

        # Add grid lines
        plt.gca().set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        plt.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        plt.title(f"Step {step}: Position ({x}, {y})")

        # Dynamic display
        plt.pause(0.5)  # Pause 0.5s per step

        # Pause 2s when reaching goal
        if (x, y) == env.goal:
            plt.pause(2)
            break

    plt.show()


def preview_maze(env):
    """Display the initial maze layout in a style consistent with the animation"""
    plt.figure(figsize=(8, 8))
    grid = np.zeros((env.size, env.size))

    # Set elements
    for (x, y) in env.obstacles:
        grid[x, y] = 1  # Obstacle (black)
    grid[env.start] = 2  # Start (green)
    grid[env.goal] = 3  # Goal (red)
    grid[env.key_pos] = 4  # Key (gold)

    # Use the same colormap as in animate_path
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'gold'])
    plt.imshow(grid.T, cmap=cmap)

    # Add grid lines (same style as animate_path)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend (matching the style of animate_path)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='green', label='Start'),
        plt.Rectangle((0, 0), 1, 1, fc='red', label='Goal'),
        plt.Rectangle((0, 0), 1, 1, fc='gold', label='Key'),
        plt.Rectangle((0, 0), 1, 1, fc='black', label='Obstacle'),
        plt.Rectangle((0, 0), 1, 1, fc='white', label='Passable')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("Initial Maze Layout Preview (Close window to start training)")
    save_path="C:\\Users\\zhang\\Desktop\\Q-learning走迷宫\\save_map"
    os.chdir(save_path)
    plt.savefig("map.png", bbox_inches='tight', dpi=300)
    plt.show()


def train():
    env = Maze(size=10)
    preview_maze(env)
    agent = QLearningAgent(env.size, env.size)
    memory = PrioritizedReplayBuffer(10000)

    best_path = None
    best_reward = -float('inf')
    max_steps = 400  # Maximum steps per episode
    iteration_num = 1000

    # New: Initialize statistics variables
    episode_rewards = []  # Record total reward per episode
    success_rates = []  # Record cumulative success rate
    success_steps = []  # Record steps for successful episodes (0 for failures)
    success_count = 0  # Success counter

    for episode in range(iteration_num):
        state = env.reset()
        done = False
        current_path = [state]
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps:
            action = agent.get_action(state, has_key=env.has_key)
            next_state, done = env.step(action)

            reward = get_reward(state, next_state, done, env, step_count, max_steps)
            memory.add(state, action, reward, next_state, done)
            current_path.append(next_state)
            total_reward += reward
            step_count += 1

            if len(memory) >= 32:
                indices, batch, weights = memory.sample(32)
                td_errors = agent.learn(batch, env)
                memory.update_priorities(indices, td_errors)

            state = next_state

        # New: Record statistics
        episode_rewards.append(total_reward)
        if done:
            success_count += 1
            success_steps.append(step_count)
        else:
            success_steps.append(0)  # Failed episodes marked as 0
        success_rates.append(success_count / (episode + 1) * 100)  # Calculate cumulative success rate

        Is = (total_reward > best_reward)
        if Is and done:
            best_reward = total_reward
            best_path = current_path
            print(f"Episode {episode}: Best path! Steps: {len(best_path) - 1}, Reward: {best_reward:.1f}")

        if episode % 100 == 0:
            print(f"Episode {episode}")

    # Visualize after training
    animate_path(env, best_path)

    # New: Plot final training statistics
    plt.figure(figsize=(15, 5))

    # Subplot 1: Smoothed rewards per episode
    plt.subplot(1, 3, 1)
    window_size = 50
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(smoothed_rewards, color='blue')
    plt.title('Smoothed Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Success rate
    plt.subplot(1, 3, 2)
    plt.plot(success_rates, color='green')
    plt.title('Success Rate Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Steps to success (successful episodes only)
    plt.subplot(1, 3, 3)
    valid_steps = [s for s in success_steps if s > 0]
    if valid_steps:
        plt.plot(valid_steps, 'ro', markersize=4, alpha=0.5, label='Individual Runs')
        # Calculate moving average
        avg_window = min(50, len(valid_steps))
        avg_steps = np.convolve(valid_steps, np.ones(avg_window) / avg_window, mode='valid')
        plt.plot(avg_steps, 'r-', linewidth=2, label=f'{avg_window}-episode Avg')
        plt.legend()
        plt.title('Steps to Success\n(Successful Episodes Only)')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()