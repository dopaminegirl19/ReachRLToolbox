import numpy as np

## HYPERPARAMETERS
ACTION_WEIGHT = 0.5 # effectuates momentum; 
TIME_LIMIT = 30     # max time steps per trial 

class ForceField():
    
    def __init__(self, start_pos = (.5, 1), goal_top = 0, goal_left = .3, goal_right = .7, goal_bottom = -1):
        """Initialize forcefield environment.
        """
        
        self.action_size = 2 # (x_velocity, y_velocity)
        self.start_pos = start_pos 
        self.goal_top = goal_top
        self.goal_left = goal_left
        self.goal_right = goal_right
        self.goal_bottom = goal_bottom
        
    def reset(self, pos=(.5, 1)):
        """Reset the environment to the starting position. The start position is (1, .5) on a 2D coordinate system). 
        """
        
        self.pos = pos
        self.state = np.array(list(pos) + [0, 0]) # State is [x_position, y_position, x_velocity, y_velocity].  
        self.next_state = None
        self.reward = 0
        self.done = 0
        self.time = 0
        
        return self 
        
    def step(self, action, time_limit=TIME_LIMIT):
        """Agent acts in the environment and gets the resulting next state and reward obtained.
        """
        
        # Add time:
        self.time += 1
        
        # Calculate new velocities by weighting with old actions: 
        old_action = self.state[2:]
        x_vel, y_vel = get_carried_action(old_action, action)

        # Calculate new positions by adding new velocities to position 
        x_pos = self.state[0] + x_vel
        y_pos = self.state[1] + y_vel
        
        # Update position and state
        self.pos = (x_pos, y_pos)
        self.state = np.array([x_pos, y_pos, x_vel, y_vel])
        
        # Check if finished 
        if self.goal_left <= self.pos[0] and self.goal_right >= self.pos[0]:         # reached goal in x dimension 
            if self.goal_top >= self.pos[1] and self.goal_bottom <= self.pos[1]:     # reached goal in y dimension
                self.reward = 1 - np.linalg.norm(action, 2)
                self.done = True 
        elif self.time >= TIME_LIMIT:     # reached time limit
            self.reward = 0 - np.linalg.norm(action, 2)
            self.done = True 
        else:                             # not finished 
            self.reward = 0 - np.linalg.norm(action, 2)
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
                           
                           
        