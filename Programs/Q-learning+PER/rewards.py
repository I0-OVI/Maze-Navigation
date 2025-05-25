def get_reward(state, next_state, done, maze):
    if done:
        return 10.0  # Reward for reaching the goal
    elif next_state in maze.obstacles:
        return -5.0  # Penalty for hitting a wall
    else:
        # Calculate Manhattan distance to goal
        old_dist = abs(state[0] - maze.goal[0]) + abs(state[1] - maze.goal[1])
        new_dist = abs(next_state[0] - maze.goal[0]) + abs(next_state[1] - maze.goal[1])

        if new_dist < old_dist:  # Moving closer to goal
            return 0.3
        elif new_dist > old_dist:  # Moving away from goal
            return -0.5
        else:
            return -0.1  # Penalty for staying in place