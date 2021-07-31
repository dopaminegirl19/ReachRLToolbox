import numpy as np

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
        
        self.state = pos + (0, 0) # State is 4 tuple (x_position, y_position, x_velocity, y_velocity).  
        return self 
        
    def act(self, action):
        """Agent acts in the environment and gets the resulting next state and reward obtained.
        """
        
        
        pass
        # return env_info.next_state, env_info.reward, env_info.done
        