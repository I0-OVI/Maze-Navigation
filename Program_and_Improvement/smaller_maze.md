# Improvements

When I got the zip file provided by wzq, I run the program without any testing. 
As for the 5*5 and 6*6 maze, the agent could solve the maze in short time. However, the agent was **stuck for prolong time** when solving the 8*8 maze.
AI gave me a lot of suggestions especially about the reward function. Although many changes were implemented, the training process was literally 'stopped' after finishing one episode.
Therefore, I thought I should have a new program to solve the maze.
### Contents
1. Introduction of PER
2. Implement of PER

#### Introduction of PER
PER stands for **Prioritized Experience Replay**. 
The principle of it is to take TD errors as a prioritized standard to automatically adjust the frequency of usage of sampling in order to accelerating the convergence of Q-values.
It aims to reduce low-value actions: collision to the wall. As the result, the valued actions will not be ignored and the efficiency can be improved.
Think of PER as a randomized dijkstra: Dijsktra skips the useless path by implementing the priority queue, PER ignores the low-value samples. But, one thing needs to be mention: dijkstra algorithm always has certain choice on the side picked. PER maintains the randomness to adapt to the uncertainty of environment.\
Here are the brief steps of PER：
1. store the sampling after taking action *a* under condition *s* and calculate the priority of this 'experience'.
2. use **Sumtree** (a binary tree stores data via recording priority of son roots in father roots) to store the samplings and choose one of them by generating a random number.
3. calculate the **TD errors** and update the Q-value.
4. use the bias correction due to the structure change by implementing the prioritized sampling.

#### Implement of PER
As we have introduced the principle of the PER, let's fast forward to the program modification.
The first part is to modify the train() function:
```python
def train():
    env = Maze(size=10)
    agent = QLearningAgent(env.size, env.size)
    memory = PrioritizedReplayBuffer(10000)

    best_path = None
    best_steps = float('inf')

    for episode in range(500):
        state = env.reset()
        done = False
        current_path = [state]  # Track current path

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

            if len(current_path) > 100:  # Prevent infinite loops
                break

        # Update best path
        if done and len(current_path) < best_steps:
            best_steps = len(current_path)
            best_path = current_path
            print(f"New best path! Steps: {best_steps}, Episode: {episode}")

    # Visualization after training
    print("\nTraining complete! Animating the best path...")
    animate_path(env, best_path)
```

Due to the charge of cursor, I used deepseek assisting me to complete the program. Therefore, I did not maintain the visualization function of original program because too much requirement may lead deepseek to return "server is busy" and refuse my requests(But in the later part, I think I would add this function to the new program to analyze problem occurred).

Second part is the addition of [memory.py]() to implement the **SumTree** feature, which is the crucial part of PER algorithm. \

Third part is the complete change in the [agent.py]() which reduced memory burden and simplified the whole program.
Here is an example:
'''
self.q_table = np.zeros((maze_size_x, maze_size_y, action_size)) #new one
self.q_table = {}  													#original one
'''
The Q-table in the original program may have dynamic increment of size but pre-allocation of numpy array was implemented to reduce the access time which was a better choice for the maintenance of smaller maze.

By the way, the reward function is regenerated. Here is the code:
```python
def get_reward(state, next_state, done, maze):
    if done:
        return 10.0  # reach the destination
    elif next_state in maze.obstacles:
        return -5.0  # penalty when collision between wall
    else:
        # reward based on the distance
        old_dist = abs(state[0] - maze.goal[0]) + abs(state[1] - maze.goal[1])
        new_dist = abs(next_state[0] - maze.goal[0]) + abs(next_state[1] - maze.goal[1])

        if new_dist < old_dist:  # move towards the destination
            return 0.3
        elif new_dist > old_dist:  # move away from the destination
            return -0.5
        else:
            return -0.1  # penalty if does not move
```

These are the brief introduction of the PER modification. I realized that Q-learning should be more powerful than I thought because navigating the maze can be quickly solved by 
A* algorithm. In the next page, I will add a task to the agent: once the agent completes the task, It is allowed to go to the destination.