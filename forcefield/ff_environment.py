import numpy as np

class ForceField():
    
    def __init__(self):
        """Initialize forcefield environment.
        """
        
        self.action_size = 2 # velocity in 2D space 
        
    def reset(self, pos=(1, .5)):
        """Reset the environment to the starting position. The start position is (1, .5) on a 2D coordinate system). 
        """
        
        self.state = pos
        return self 
        
    def act(self, action):
        """Agent acts in the environment and gets the resulting next state and reward obtained.
        """
        
        
        pass
        # return env_info.next_state, env_info.reward, env_info.done
        