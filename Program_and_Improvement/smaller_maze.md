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
Taking an analogy, the addition of PER is like the dijkstra having the priority queue optimization
