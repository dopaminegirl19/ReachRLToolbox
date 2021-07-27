# ReachRLToolbox

**TestRL.ipynb** is a notebook that investigates the feasibility of using RL agent (qlearning in this case) to model humans reaching movements. This notebook investigates the following experimental paradigms using a toy example : 
* Simple reaching to a single target
* Reaching to a redundant target (ie. rectangular target)
* Multiple targets
* Obstacles in the environment
* Multiple targets with different rewards

The *conclusions* from this first notebook are:

* All these experimental paradigms can be modelled using RL agent in a simple way
* A lookout table with a finite set of states and actions is not enough

# TODO list
**Test network implementation**
- [] Implemnet a NFQ to test the behaviour on the same grid worlds
**Infinite state and actions spaces**
- [] Implement a model of the system dynamics for a given state,action pair
- [] Implement a DQN network for learning the value function
- [] Eventually expand to DDPG to mimic humans behaviour
