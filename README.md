# dstar-lite-cpp
Simple D* Lite extension module written in c++ for Python 3


## What is it
Simple D* Lite implementation.
I was working on a robotics project and we used D* Lite as planning algorithm. The vanilla python implementation (inspired by https://github.com/GuanyaShi/CS133b-D-Lite-Simulation) delivered usatisfactory performance, so I've written this extension module to exploit the speed of c++. I have no benchmark, but I run some empirical test. With my current setup (intel core i7 6500U, 8GB RAM), the python implementation took almost 8 hours to solve about 10000 randomly generated mazes. On the contrary, the same task was carried out within 2 hours using this extension module.

This customization allow the user to specifiy a constaint in term of how far the resulting path should be with respect to any obstacle (e.g. *obstacle margin*). Even if the requested goal is not reacheable with current settings, the algorithm should try to drive you as close to the destination as possible.

Disclaimer: use this code in production at your own risk.