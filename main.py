import numpy as np
# from cpp_dstar_lite import DStarLite
from cpp_dstar_lite import DStarLite

h, w = 100, 100
map = np.where(np.random.rand(h, w) > 0.95, np.inf, 0)
start = np.array((99, 50))
goal = np.array((0, 50))
map[max(0, goal[0] - 1):goal[0] + 2, max(0, goal[1] - 1):goal[1] + 2] = np.inf
ds = DStarLite(map, goal[0], goal[1], start[0], start[1], 10000, False, 1, 1)
next_step = start
path = []
while next_step is not None:
    next_step = ds.step()
    next_step = ds.step(map)
    path.append(next_step)
    print(next_step)
pass