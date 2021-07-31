import numpy as np

## HYPERPARAMETERS
ACTION_WEIGHT = 0.5 # effectuates momentum; 

class ForceField():
    
    def __init__(self, start_pos = (.5, 1), goal = (.5, 0)):
        """Initialize forcefield environment.
        """
        
        self.action_size = 2 # (x_velocity, y_velocity)
        self.start_pos = start_pos 
        self.goal = goal
        
    def reset(self, pos=(.5, 1)):
        """Reset the environment to the starting position. The start position is (1, .5) on a 2D coordinate system). 
        """
        
        self.pos = pos
        self.state = list(pos) + [0, 0] # State is 4 tuple (x_position, y_position, x_velocity, y_velocity).  
        self.next_state = None
        self.reward = 0
        self.done = 0
        
        return self 
        
    def step(self, action):
        """Agent acts in the environment and gets the resulting next state and reward obtained.
        """
        
        # Calculate new velocities by weighting with old actions: 
        old_action = self.state[2:]
        x_vel, y_vel = get_carried_action(old_action, action)

        # Calculate new positions by adding new velocities to position 
        x_pos = self.state[0] + x_vel
        y_pos = self.state[1] + y_vel
        
        # Update position and state
        self.pos = (x_pos, y_pos)
        self.state = [x_pos, y_pos, x_vel, y_vel]
        
        # Check if finished 
        if self.pos == self.goal:
            self.reward = 0.1
            self.done = True 
        else:
            self.reward = -0.1
            self.done = False 
        
        return self
                           
def get_carried_action(old_action, action, act_weight = ACTION_WEIGHT):
        """Get new action based on action from previous timestep. 
        Carried action = (action_weight * old_action) + new_action 
        Params
        ======
        old_action: list length 2 [old_x_velocity, old_y_velocity]
        action: list length 2 [new_x_velocity, new_y_velocity]
        act_weight: weight applied to old velocity 
        """
        
        new_x_velocity = old_action[0] * act_weight + action[0]
        new_y_velocity = old_action[1] * act_weight + action[1]
        
        return new_x_velocity, new_y_velocity 
                           
                           
        