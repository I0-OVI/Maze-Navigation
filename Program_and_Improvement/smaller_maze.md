# Improvements

When I got the zip file provided by wzq, I run the program without any testing. 
As for the 5*5 and 6*6 maze, the agent could solve the maze in short time. However, the agent was **stuck for prolong time** when solving the 8*8 maze.
AI gave me a lot of suggestions especially about the reward function. Although many changes were implemented, the training process was literally 'stopped' after finishing one episode.
Therefore, I thought I should have a new program to solve the maze.

### Contents
1. Introduction of PER
2. Implement of PER
3. Debugging

#### Introduction of PER
PER stands for **Prioritized Experience Replay**. 
The principle of it is to take TD errors as a prioritized standard to automatically adjust the frequency of usage of sampling in order to accelerating the convergence of Q-values.
It aims to reduce low-value actions: collision to the wall. As the result, the valued actions will not be ignored and the efficiency can be improved.
Think of PER as a randomized dijkstra: Dijsktra skips the useless path by implementing the priority queue, PER ignores the low-value samples. But, one thing needs to be mention: dijkstra algorithm always has certain choice on the side picked. PER maintains the randomness to adapt to the uncertainty of environment.\
Here are the brief steps of PER：
1. store the sampling after taking action *a* under condition *s* and calculate the priority of this 'experience'
2. use **Sumtree** (a binary tree stores data via recording priority of son roots in father roots) to store the samplings and choose one of them by generating a random number
3. calculate the **TD errors** and upadate the Q-value
4. use the bias correction due to the structure change by implementing the prioritized sampling

#### Implement of PER
