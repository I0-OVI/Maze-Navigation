def get_reward(state, next_state, done, maze, step_count, max_steps=50):
    """
    Improved reward function:
    - Add step penalty
    - Guide in stages
    - Prevent going in circles
    """
    # Significantly increase the penalty for staying in the same place
    if state == next_state:
        return -10

    if done:
        # Success reward + step reward (the earlier the completion, the higher the reward)
        step_bonus = (1 - step_count / max_steps) * 5
        return 10.0 + step_bonus

    # Reward for obtaining the key (first time)
    if next_state == maze.key_pos and not maze.has_key:
        return 5.0

    # Penalty for invalid actions
    if next_state in maze.obstacles:
        return -5.0  # Hitting a wall
    if next_state == maze.goal and not maze.has_key:
        return -3.0  # Trying to pass the goal without a key

    # Step penalty (small penalty for each step)
    step_penalty = -0.1

    # Directional guidance reward
    if not maze.has_key:
        target = maze.key_pos
        move_reward = 1.8
    else:
        target = maze.goal
        move_reward = 2

    # Calculate the change in distance
    old_dist = maze.bfs_distance(state, target, include_key=True)
    new_dist = maze.bfs_distance(next_state, target, include_key=True)

    # Combined reward
    if new_dist < old_dist:
        return move_reward + step_penalty
    elif new_dist > old_dist:
        return -1.8 + step_penalty
    else:
        return -10 + step_penalty