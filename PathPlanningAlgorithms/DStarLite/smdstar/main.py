import gymnasium as gym
from grid import GridEnv

# Create a Grid Environment with some obstacles
env = GridEnv(grid_size=(60, 60), start=(0, 0), goal=(4, 4), obstacles=[(2, 2), (1, 3), (3, 1)])

# Reset the environment
obs = env.reset()

for step in range(50):
    env.render()
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if done:
        print("Goal reached!")
        break

env.close()
