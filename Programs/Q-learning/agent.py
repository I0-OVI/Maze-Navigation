import numpy as np
import random


class QLearningAgent:
    def __init__(self, maze, reward_system, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=0.1):
        self.maze = maze
        self.reward_system = reward_system
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self._visited = set()  # 记录访问过的状态
        self._stuck_counter = 0
        # 动作空间：上、下、左、右
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        # 初始化Q表
        self.q_table = {}

    def get_q_value(self, state, action):
        """获取Q值，如果状态不存在则初始化为0"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        return self.q_table[state][action]

    def choose_action(self, state):
        """选择动作（ε-贪婪策略）"""
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        # 选择Q值最大的动作
        max_q = max(self.q_table[state].values())
        best_actions = [a for a, q in self.q_table[state].items() if q == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        current_q = self.get_q_value(state, action)

        # 计算下一个状态的最大Q值
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q

    def take_action(self, action):
        """执行动作并返回结果"""
        dx, dy = self.action_map[action]
        current_x, current_y = self.maze.agent_position
        new_x, new_y = current_x + dx, current_y + dy

        action_result = {
            'hit_wall': False,
            'in_trap': False,
            'reached_exit': False,
            'valid_move': False
        }

        if self.maze.is_valid_position(new_x, new_y):
            action_result['valid_move'] = True
            self.maze.set_agent_position(new_x, new_y)

            if (new_x, new_y) in self.maze.trap_positions:
                action_result['in_trap'] = True
            elif (new_x, new_y) == self.maze.exit_position:
                action_result['reached_exit'] = True
        else:
            action_result['hit_wall'] = True

        return action_result