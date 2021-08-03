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

**OFC FF** is a notebook that implements the optimal feedback control model for reaching movements and investigate what would be the optimal behaviour in presence of force field (here a constant lateral force field is considered). 

This demonstrates the following result:

* The optimal trajectory that the agent reaches at the end of motor learning (ie. asymptotycaly when they have learned the environment perfectly) is not a straight line
* The optimal trajectory depends on the force field intensity
* Metrics such as the peak lateral deviation is not enough to investigate motor learning as it will consider that the optimal trajectory is the straight line path
