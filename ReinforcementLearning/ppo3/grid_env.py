import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt



class GridEnvironment(gym.Env):
    def __init__(self, sx=0, sy=0, gx=59, gy=59, ox=None, oy=None, grid_size=60):
        super(GridEnvironment, self).__init__()

        self.grid_size = grid_size
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(8)  # Change action space to 8 discrete actions

         # Define start and goal positions
        self.sx, self.sy = sx, sy
        self.gx, self.gy = gx, gy
        self.start_position = (sx, sy)
        self.goal_position = (gx, gy)

        # Define obstacle lists
        self.ox = ox if ox is not None else []
        self.oy = oy if oy is not None else []

        # Initialize grid and add obstacles
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.add_obstacles()



        # Set initial position of the agent
        self.agent_position = self.start_position

    def add_obstacles(self):
        for x, y in zip(self.ox,self.oy):
            # Avoid placing obstacles at the start or goal position
            if (x, y) != self.start_position and (x, y) != self.goal_position:
                self.grid[x, y] = 1  # 1 represents an obstacle


    def step(self, action):
        # Map discrete action to movement directions
        new_position = list(self.agent_position)

        # Define the discrete movements for each action
        if action == 0 and new_position[0] > 0:  # Up
            new_position[0] -= 1
        elif action == 1 and new_position[0] < self.grid_size - 1:  # Down
            new_position[0] += 1
        elif action == 2 and new_position[1] > 0:  # Left
            new_position[1] -= 1
        elif action == 3 and new_position[1] < self.grid_size - 1:  # Right
            new_position[1] += 1
        elif action == 4 and new_position[0] > 0 and new_position[1] < self.grid_size - 1:  # Up-right
            new_position[0] -= 1
            new_position[1] += 1
        elif action == 5 and new_position[0] > 0 and new_position[1] > 0:  # Up-left
            new_position[0] -= 1
            new_position[1] -= 1
        elif action == 6 and new_position[0] < self.grid_size - 1 and new_position[
            1] < self.grid_size - 1:  # Down-right
            new_position[0] += 1
            new_position[1] += 1
        elif action == 7 and new_position[0] < self.grid_size - 1 and new_position[1] > 0:  # Down-left
            new_position[0] += 1
            new_position[1] -= 1

        # Check if the agent hits an obstacle
        if self.grid[tuple(new_position)] == 1:
            reward = -1
            done = False
        else:
            self.agent_position = tuple(new_position)
            reward = -0.1
            done = self.agent_position == self.goal_position

        return (
            np.array(self.agent_position, dtype=np.float32),
            reward,
            done,
            {}
        )

    def reset(self, **kwargs):
        self.agent_position = self.start_position
        return np.array(self.agent_position, dtype=np.float32)

    def render(self, mode='rgb_array'):
      grid = np.full((self.grid_size, self.grid_size, 3), [46, 149, 209], dtype=np.uint8)

      # Mark the start, goal, and agent positions
      grid[self.sx, self.sy] = [0, 255, 0]  # Green for start
      grid[self.gx, self.gy] = [0, 0, 255]    # Blue for goal
      grid[self.agent_position[0], self.agent_position[1]] = [255, 0, 0]  # Red for agent
      for x, y in zip(self.ox, self.oy):
        grid[x, y] = [0, 0, 0]  # Black for obstacles

      if mode == 'rgb_array':
          return grid
      elif mode == 'human':
          plt.imshow(grid)
          plt.title(f"Agent: {self.agent_position} | Start: {self.start_position} | Goal: {self.goal_position}")
          plt.axis('off')
          plt.show()