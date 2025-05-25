from maze import Maze, create_simple_maze, create_complex_maze, create_spiral_maze
from agent import QLearningAgent
from rewards import RewardSystem
from visualizer import MazeVisualizer
import threading
import time
import matplotlib.pyplot as plt
import os
import datetime
import os
Path="C:\\Users\\zhang\\Desktop\\Q-learning走迷宫"
os.chdir(Path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_maze_choice():
    """获取用户选择的地图"""
    while True:
        print("\n请选择迷宫难度：")
        print("1. 简单迷宫 (5x5)")
        print("2. 复杂迷宫 (8x8)")
        print("3. 螺旋迷宫 (6x6)")
        choice = input("请输入选项 (1/2/3): ")

        if choice == '1':
            return create_simple_maze(), 3000  # 简单迷宫训练3000轮
        elif choice == '2':
            return create_complex_maze(), 5000  # 复杂迷宫训练5000轮
        elif choice == '3':
            return create_spiral_maze(), 4000  # 螺旋迷宫训练4000轮
        else:
            print("无效的选项，请重新选择")


def plot_training_stats(episode_rewards, success_rates, avg_steps, exploration_rates, save_dir):
    """绘制训练统计图表并保存到指定目录"""
    plt.figure(figsize=(15, 10))

    # 创建子图
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('每轮平均奖励')
    plt.xlabel('轮数')
    plt.ylabel('奖励')

    plt.subplot(2, 2, 2)
    plt.plot(success_rates)
    plt.title('成功率')
    plt.xlabel('轮数')
    plt.ylabel('成功率 (%)')

    plt.subplot(2, 2, 3)
    plt.plot(avg_steps)
    plt.title('平均步数')
    plt.xlabel('轮数')
    plt.ylabel('步数')

    plt.subplot(2, 2, 4)
    plt.plot(exploration_rates)
    plt.title('探索率变化')
    plt.xlabel('轮数')
    plt.ylabel('探索率')

    plt.tight_layout()

    # 保存图表
    plt.savefig(os.path.join(save_dir, 'training_stats.png'))
    plt.close()


def train_agent(maze, episodes=2000, visualize=False):
    """训练智能体（完整保留统计图表功能）"""
    # 初始化系统和统计变量
    reward_system = RewardSystem(maze)
    agent = QLearningAgent(maze, reward_system)
    if visualize:
        visualizer = MazeVisualizer(maze)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"

    # 训练统计数据容器（必须保留的变量）
    episode_rewards = []  # 每轮总奖励
    success_rates = []  # 累计成功率
    avg_steps_list = []  # 平均步数
    exploration_rates = []  # 探索率变化
    success_count = 0  # 成功次数计数器
    total_steps = 0  # 总步数计数器
    best_reward = float('-inf')  # 最佳奖励记录

    print(f"\n开始训练，共{episodes}轮...")
    for episode in range(episodes):
        state = maze.reset()
        episode_reward = 0
        done = False
        steps = 0

        # 重置距离记录（如果使用距离奖励）
        if hasattr(reward_system, '_last_distance'):
            del reward_system._last_distance
        print("*******")
        # 单轮训练循环
        while not done:
            action = agent.choose_action(state)
            action_result = agent.take_action(action)

            # 更新位置历史（用于防绕路检测）
            if hasattr(reward_system, 'update_position_history'):
                reward_system.update_position_history(maze.agent_position)

            reward = reward_system.get_reward(action_result)
            next_state = maze.get_state()

            agent.update_q_value(state, action, reward, next_state)
            episode_reward += reward
            state = next_state
            steps += 1
            done = action_result['reached_exit'] or action_result['in_trap']

            if visualize and steps % 5 == 0:
                visualizer.update()
                time.sleep(0.02)

        # 更新统计指标（必须保留的核心逻辑）
        total_steps += steps
        success_count += int(action_result['reached_exit'])
        best_reward = max(best_reward, episode_reward)

        # 记录关键指标（图表数据来源）
        episode_rewards.append(episode_reward)
        success_rates.append(success_count / (episode + 1) * 100)
        avg_steps_list.append(total_steps / (episode + 1))
        exploration_rates.append(agent.exploration_rate)

        # 训练进度输出
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode {episode + 1}/{episodes}:")
            print(f"最近100轮成功率: {(success_count / 100) * 100:.1f}%")
            print(f"平均步数: {total_steps / 100:.1f}")
            print(f"最高奖励: {best_reward:.1f}")
            # 重置窗口统计
            success_count = 0
            total_steps = 0
            best_reward = float('-inf')
        elif (episode + 1) % 10 == 0:
            print(".", end="", flush=True)

    # 图表保存功能（必须保留的核心部分）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    metrics = [
        ('每轮奖励', episode_rewards),
        ('成功率(%)', success_rates),
        ('平均步数', avg_steps_list),
        ('探索率', exploration_rates)
    ]

    for i, (title, data) in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(data)
        plt.title(title)
        plt.xlabel('轮数')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_stats.png'))
    plt.close()

    print(f"\n训练完成！图表已保存至: {output_dir}")
    return agent, maze, {
        'rewards': episode_rewards,
        'success_rates': success_rates,
        'avg_steps': avg_steps_list,
        'exploration_rates': exploration_rates
    }

def test_agent(agent, maze, visualize=False):
    """测试训练好的智能体"""
    state = maze.reset()
    done = False
    steps = 0

    if visualize:
        visualizer = MazeVisualizer(maze)

    print("\n开始测试训练好的智能体:")
    while not done and steps < 100:
        action = agent.choose_action(state)
        action_result = agent.take_action(action)
        state = maze.get_state()
        steps += 1

        if visualize:
            visualizer.update()  # 更新渲染
            time.sleep(0.3)  # 放慢速度便于观察

        if action_result['reached_exit']:
            print("成功到达出口!")
            break
        elif action_result['in_trap']:
            print("掉入陷阱!")
            break


if __name__ == "__main__":
    # 确保output目录存在
    os.makedirs("output", exist_ok=True)

    # 获取用户选择的地图和对应的训练轮数
    maze, episodes = get_maze_choice()

    # 训练智能体（不带可视化，因为会干扰训练过程）
    agent, maze, stats = train_agent(maze, episodes=episodes, visualize=False)

    # 测试智能体（带可视化）
    test_agent(agent, maze, visualize=True)