# Q-learning

Q-learning is a model-free reinforcement learning algorithm where the **agent** learns to make decisions by interacting with the **environment**.
Its main process involves maintance of Q-table which contains many Q-values gained from an evaluation function of the agent's motion and find the best strategy of the problem.\
Here is the state transition equation of Q-values:\
$Q(s,a)=Q(s,a)+\gamma \sum_{s'} P(s'| s,a) V(s')$\

In the equation, the explanation of varible is as follows:\
$Q(s,a)$: Q-value of taking action *a* in state *s* \
a: rate of learning (bewteen 0(learn nothing) and 1(fully learn)) \
$\gamma$: discount factor (0(only care about current reward)~1(prioritize long-term reward)) \
r: reward recieved after taking action *a* in state *s* \
s': the next state transitioned to after taking action *a* in state *s*

There is a thing needed to mention: the basic principle of Q-learning is not brute-force search. 
Instead, DP, TD and greedy occupies the most. 
### In maze
An intersting thing about this algorthim used in maze: 
the program only oberves the current position of the agent and they must take action to know if it was in the trap or has wall collision, this step seems to be brute-force search
