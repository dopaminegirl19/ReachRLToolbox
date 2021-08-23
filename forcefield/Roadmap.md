## Baseline ##

The origin branch repo should have this structure:

```
README.md
Roadmap.md
ForceField.ipynb
BehaviorName02.ipynb
BehaviorName03.ipynb
ff_environment.py
agents
└───ddpg-32-16
│   │   learned-ForceField.pth
│   │   learned-BehaviorName02.pth
└───ddpg-64-32
│   │   learned-BehaviorName03.pth
models
│   1-layer
│   2-layer
│   3-layer
```

To do: decide how OFC work fits in to directory... 


## Guides for new behaviors ##

Each behavior should be documented in a .ipynb notebook. A behavior corresponds to an environment, and all environments are buildable from ff_environment.py. 
Therefore, to develop a new behavior:

1. Create a new branch named with your behavior (e.g. NarrowTarget)
1. In the branch, you can modify ff_environment and create new agents and model architectures at will. New agents and models are saved separately. Note that model.py 
denotes the depth of the network; agent.py denotes the widths of the layers. 
1. Do the work, and the ultimate goal is a behavior-n.ipynb notebook where you document and visualize your environment, agent, etc and the agent's behavior. 
1. You should also be sure you can recreate the existing behavior notebooks if you have made any modifications to ff_environment.py (and even if you haven't, just to be sure).
1. You can now merge the branch to the origin. 

For now, there is a *one to one mapping of behavior to agent*. The agent used is explicitly specified in the behavior notebook. 
