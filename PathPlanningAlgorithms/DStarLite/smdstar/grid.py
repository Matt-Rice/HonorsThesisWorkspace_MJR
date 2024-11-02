import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.agent_pos = np.array(self.start)

        # Create a 2D grid, initially all free (0: free space, 1: obstacle)
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # Add obstacles (if any)
        if obstacles:
            for obs in obstacles:
                self.grid[obs] = 1  # Set obstacles to 1

        # Define action space (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)

        # Define observation space (grid state + agent position)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=self.grid_size[0] - 1, shape=(2,), dtype=np.int32),
            "grid": spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.int32),
        })

        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.agent_pos = np.array(self.start)
        return {"agent": self.agent_pos, "grid": self.grid.copy()}

    def step(self, action):
        """Take a step in the environment based on action."""
        # Define movement for the agent (up, down, left, right)
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = moves[action]

        # Compute new position
        new_pos = self.agent_pos + np.array(move)

        # Check boundaries
        if (0 <= new_pos[0] < self.grid_size[0]) and (0 <= new_pos[1] < self.grid_size[1]):
            # Check for obstacles
            if self.grid[tuple(new_pos)] == 0:
                self.agent_pos = new_pos

        # Check if goal is reached
        done = np.array_equal(self.agent_pos, self.goal)
        reward = 1 if done else -0.1  # Reward for reaching the goal, small penalty otherwise

        return {"agent": self.agent_pos, "grid": self.grid.copy()}, reward, done, False, {}

    def render(self, mode="human"):
        """Render the current grid and agent position."""
        grid_to_render = self.grid.copy()
        grid_to_render[tuple(self.agent_pos)] = 2  # Mark agent as '2'
        grid_to_render[self.goal] = 3  # Mark goal as '3'
        print(grid_to_render)

    def close(self):
        pass
