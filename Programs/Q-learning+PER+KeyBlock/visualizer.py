import pygame
import numpy as np
from maze import Maze


class MazeVisualizer:
    def __init__(self, maze, cell_size=80):
        self.maze = maze
        self.cell_size = cell_size
        self.window_width = maze.width * cell_size
        self.window_height = maze.height * cell_size

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SHOWN
        )
        pygame.display.set_caption("Q-learning Maze")
        pygame.display.set_allow_screensaver(False)

        self.colors = {
            'wall': (100, 100, 100),
            'trap': (255, 0, 0),
            'exit': (0, 255, 0),
            'agent': (0, 0, 255),
            'floor': (200, 200, 200),
            'background': (255, 255, 255)
        }

    def draw_cell(self, x, y, color):
        rect = pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw_maze(self):
        self.screen.fill(self.colors['background'])
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                self.draw_cell(x, y, self.colors['floor'])

        for x, y in self.maze.wall_positions:
            self.draw_cell(x, y, self.colors['wall'])

        for x, y in self.maze.trap_positions:
            self.draw_cell(x, y, self.colors['trap'])

        if self.maze.exit_position:
            x, y = self.maze.exit_position
            self.draw_cell(x, y, self.colors['exit'])

        if self.maze.agent_position:
            x, y = self.maze.agent_position
            self.draw_cell(x, y, self.colors['agent'])

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def render(self):
        self.draw_maze()
        pygame.display.flip()
        pygame.time.wait(10)

    def update(self):
        self.render()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.render()
        pygame.quit()