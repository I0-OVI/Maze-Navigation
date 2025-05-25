# Task maze

As I have said the additional task would be implemented to the agent: the agent needs to first reach a specific block to get the 'key'. Once the agent gets 'key', it could go to the destination or the reach of target could be taken as unsuccessful access.
(image)
In the picture, the agent should first go to the yellow block and then navigate to the red block. \
There are some changes taken place in the new program in maze.py which is responsible for the generation of the maze.

```python
def step(self, action):
    x, y = self.state
    dx, dy = self.action_effects[action]
    new_x, new_y = x + dx, y + dy

	#check whether the motion is available 
    if (0 <= new_x < self.size and 0 <= new_y < self.size and
        (new_x, new_y) not in self.obstacles):
        self.state = (new_x, new_y)

        # check if having the key
        if self.state == self.key_pos:
            self.has_key = True

    # desination check(can be accessed only with key)
    done = (self.state == self.goal) and self.has_key
    return self.state, done
```
This is the check for having a key
```python
def _path_exists(self, start, end, obstacles):
    visited = set()
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            return True
        if (x, y) in visited or (x, y) in obstacles:
            continue
        visited.add((x, y))
        # use the initialized 'action_effects'
        for dx, dy in self.action_effects:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
            queue.append((nx, ny))
    return False
```
I add a function using BFS to check whether the pass is available in case of that agent cannot reach the destination in maze due to the random generation of maze.The rest parts have several changes and I am too lazy to list all of them. This folder involves all the code in this section.

### Debugging and Improvement 

**BFS distance check** \
An animated function was implement, every step of the agent could be observed. If the destination is rounded by the wall in the picture, the agent would be mislead by the reward function using the direct path calculation ($D=((x_1-x_2)^2+(y_1-y_2))^-2$) and have a meaningless circle motion.
```python
    def bfs_distance(self, start, end, include_key=False):
        """calculate the shortest distance considering the obstacle"""
        if start == end:
            return 0

        visited = set()
        queue = deque([(start, 0)])  # (position, distance)

        while queue:
            (x, y), dist = queue.popleft()

            # check if reaching the destination
            if (x, y) == end:
                return dist

            # skip the accessed obstacle
            if (x, y) in visited or (x, y) in self.obstacles:
                continue

            visited.add((x, y))

            # check if the agent have key
            if include_key and (x, y) == self.key_pos:
                continue  

            # explore in four direction
            for dx, dy in self.action_effects:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    queue.append(((nx, ny), dist + 1))

        return float('inf')  # unreachable
```

**Score grinding** \
Literally, the agent did this ridiculous thing because I forgot to give a requirement to deepseek when it is generating the path updating part.
```python
#	original one
if total_reward > best_reward and done: 		

#	improved one
Is = (best_path is None) or len(current_path) - 1 < len(best_path) - 1 or (current_path == best_reward and total_reward > best_reward)
if Is and done:
```
The statics graph can be a clearer visual demonstration:
(image_2)
The picture above shows the pre-modification state, the following graph was generated after modification.
(image_3)


The episode before 500 involves some negative rewards because I have set extreme heavy penalty for staying original place and moving in circle.
The difference can be seen in the last episodes: the unmodified agent returns an increasing reward and huge number of steps. In contrast, the improved one has a decreasing trend in the end. Plus, the final penalty was reduced a lot. \
However, I was not satisfied with this result due to the large number of steps. Therefore, the first thing came to my mind was to adjust the value in the reward function. I would say this function only contribute a little to the final result. Later, I suspected whether the program for agent had some issues. Deepseek thought I was right and gave me a new *agent.py*, pointing out that the agent did not update the state even if it had got the key. As the result, the statics graph is pretty good and animation of path was reasonable.
(image_1)