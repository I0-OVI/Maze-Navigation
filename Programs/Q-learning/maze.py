import numpy as np


class Maze:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 0表示空地
        self.wall_positions = set()  # 墙体位置
        self.trap_positions = set()  # 陷阱位置
        self.exit_position = None  # 出口位置
        self.agent_position = None  # 智能体位置

    def add_wall(self, x, y):
        """添加墙体"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.wall_positions.add((x, y))
            self.grid[y, x] = 1  # 1表示墙体

    def add_trap(self, x, y):
        """添加陷阱"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.trap_positions.add((x, y))
            self.grid[y, x] = 2  # 2表示陷阱

    def set_exit(self, x, y):
        """设置出口"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.exit_position = (x, y)
            self.grid[y, x] = 3  # 3表示出口

    def set_agent_position(self, x, y):
        """设置智能体位置"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.agent_position = (x, y)

    def is_valid_position(self, x, y):
        """检查位置是否有效（不超出边界且不是墙体）"""
        return (0 <= x < self.width and
                0 <= y < self.height and
                (x, y) not in self.wall_positions)

    def get_state(self):
        """获取当前状态"""
        return self.agent_position

    def reset(self):
        """重置迷宫"""
        self.agent_position = (0, 0)  # 默认从左上角开始
        return self.get_state()


def create_simple_maze():
    """创建优化后的简单迷宫 (5x5)"""
    maze = Maze(5, 5)
    # 墙体布局，留出一条通路
    maze.add_wall(1, 0)
    maze.add_wall(1, 1)
    maze.add_wall(3, 1)
    maze.add_wall(3, 2)
    maze.add_wall(1, 3)
    maze.add_wall(3, 3)
    maze.add_wall(1, 4)
    # 陷阱布局，避免完全封死
    maze.add_trap(2, 3)
    # 出口
    maze.set_exit(4, 4)
    return maze


def create_complex_maze():
    """创建8x8复杂迷宫"""
    maze = Maze(8, 8)
    # 外圈墙体
    for x in range(8):
        maze.add_wall(x, 0)
        maze.add_wall(x, 7)
    for y in range(1, 7):
        maze.add_wall(0, y)
        maze.add_wall(7, y)
    # 内部墙体
    for y in range(1, 6):
        maze.add_wall(2, y)
    for x in range(2, 6):
        maze.add_wall(x, 2)
    for y in range(2, 7):
        maze.add_wall(5, y)
    # 留出通路
    maze.wall_positions.discard((2, 4))  # 打开一个口
    maze.wall_positions.discard((5, 5))  # 打开一个口
    # 陷阱
    maze.add_trap(3, 3)
    maze.add_trap(4, 4)
    maze.add_trap(6, 5)
    # 出口
    maze.set_exit(6, 6)
    print("*******")
    return maze


def create_spiral_maze():
    """创建6x6螺旋迷宫"""
    maze = Maze(6, 6)
    # 墙体布局，螺旋形状
    wall_coords = [
        (4, 0),
        (1, 1), (2, 1),
        (2, 2), (4, 2),
        (1, 3), (3, 3), (4, 3),
        (4, 4),
        (4, 5)
    ]
    for x, y in wall_coords:
        maze.add_wall(x, y)
    # 起点S
    maze.set_agent_position(0, 5)
    # 终点
    maze.set_exit(5, 5)
    return maze