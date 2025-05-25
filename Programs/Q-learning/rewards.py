class RewardSystem:
    def __init__(self, maze):
        self.maze=maze
        # 根据迷宫大小调整奖励值
        if maze.width <= 5 or maze.height <= 5: # 简单迷宫
            self.rewards = {
                'exit': 150,  # 到达出口（适当提高）
                'trap': -60,  # 掉入陷阱（适当加重）
                'wall': -8,  # 撞墙（适当减轻）
                'step': -2,  # 每走一步（适当加重）
                'invalid': -15  # 无效移动（适当减轻）
            }
        else:  # 复杂迷宫
            self.rewards = {
                'exit': 100,  # 到达出口
                'trap': -20,  # 掉入陷阱
                'invalid': -5,  # 撞墙/无效移动
                'step': -1,  # 每步基础惩罚
                'dist_factor': 0.5  # 距离奖励系数
            }
        self.rewards.update({
            'explore_bonus': 1.8,  # 探索新区域奖励
            'loop_penalty': 4,  # 绕路惩罚
            'max': 10,  # 单步奖励上限
            'min': -5  # 单步惩罚下限
        })
        # 新增状态跟踪
        self._visited_positions = set()
        self._position_history = []

    def _is_new_area(self, pos):
        """检查是否探索新区域"""
        if pos not in self._visited_positions:
            self._visited_positions.add(pos)
            return True
        return False

    def _is_repeating_path(self):
        """检测是否在绕路（最近5步有重复位置）"""
        hist = self._position_history[-10:]  # 检查最近10步
        return len(hist) != len(set(hist))  # 有重复位置

    def update_position_history(self, pos):
        """更新位置历史（需在get_reward外部调用）"""
        self._position_history.append(pos)
        if len(self._position_history) > 20:  # 限制历史长度
            self._position_history.pop(0)
    def _manhattan_distance(self, pos1, pos2):
        """计算两个坐标之间的曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_reward(self, action_result):
        # 基础奖励
        if action_result['reached_exit']:
            return 100 + 5 * self.maze.width  # 动态终点奖励

        if action_result['in_trap']:
            return -30

        # 距离引导奖励（曼哈顿距离）
        current_pos = self.maze.agent_position
        exit_pos = self.maze.exit_position
        if exit_pos is None:
            return -1  # 无出口时默认惩罚

        dist = abs(current_pos[0] - exit_pos[0]) + abs(current_pos[1] - exit_pos[1])
        dist_reward = -0.3 * dist  # 距离惩罚

        # 探索奖励（鼓励访问新区域）
        if not hasattr(self, '_visited'):
            self._visited = set()
        explore_bonus = 1.5 if current_pos not in self._visited else 0
        self._visited.add(current_pos)

        return dist_reward + explore_bonus