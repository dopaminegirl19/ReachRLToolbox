import numpy as np
import torch
## HYPERPARAMETERS
ACTION_WEIGHT = 0.5 # effectuates momentum; 
TIME_LIMIT = 50     # max time steps per trial 
COST_PARAM = 0.002 # penalty to action - HAD TO CHANGE THAT OTHERWISE NO MOVEMENT

class Workspace():
    
    def __init__(self, start_pos = (.5, 1), goal_tl = (.1, 0), goal_br = (.9, -.8), space_padding = 5, max_len=TIME_LIMIT):
        """Initialize forcefield environment.
        Params
        ======
        start_pos: tuple (x, y) coordinates of starting position.
        goal_tl: tuple(x, y) coordinates of top left corner of goal box.
        goal_br: tuple(x, y) coordinates of bottom right corner of goal box.
        space_padding: float space from start_pos in each direction delineating allowed movement.
        """
        
        self.action_size = 2 # (x_velocity, y_velocity)
        self.max_len = max_len
        
        # goal box
        self.start_pos = start_pos 
        self.goal_left = goal_tl[0]
        self.goal_top = goal_tl[1]
        self.goal_right = goal_br[0]
        self.goal_bottom = goal_br[1]
        
        # workspace boundaries
        self.max_top = start_pos[1] + space_padding
        self.max_left = start_pos[0] - space_padding
        self.max_bottom = start_pos[1] - space_padding
        self.max_right = start_pos[0] + space_padding
        
        # forcefields
        self.ff_force = (0, 0)
        self.ff_top = self.max_top # it should cover the whole workspace
        self.ff_bottom = self.max_bottom
        
        
    def add_forcefield(self, force, top=None, bottom=None):
        """Add a forcefield. Default is to apply forcefield from a limit in the y dimension.
        Params
        ======
        force: (x, y) tuple of forces applied in each direction. 
        """
        # THIS FUNCTION IS NOT USED YET ?
        self.ff_force = force
        if top is not None:
            self.ff_top = top
        if bottom is not None:
            self.ff_bottom = bottom
        
    def reset(self, pos=(.5, 1)):
        """Reset the environment to the starting position. The start position is (1, .5) on a 2D coordinate system). 
        """
        
        self.pos = pos
        self.state = np.array(list(pos) + [0, 0, 0, 0]) # State is [x_position, y_position, x_velocity, y_velocity].
        self.next_state = None
        self.reward = 0
        self.done = 0
        self.time = 0
        
        return self 
        
    def step(self, action, act_weight = ACTION_WEIGHT, cost = COST_PARAM):
        """Agent acts in the environment and gets the resulting next state and reward obtained.

        The system dynamics comes from Nashed et al. 2012
        """
        
        # Add time (in time steps of 10 ms):
        self.time += 1
        dt, kv, tau = 0.01, 1, 0.04

        # Calculate new state using the Newtonian dynamics
        x_pos = self.state[0] + self.state[2]*dt
        y_pos = self.state[1] + self.state[3]*dt
        x_vel = (1-kv*dt) * self.state[2] + dt * self.state[4]
        y_vel = (1-kv*dt) * self.state[3] + dt * self.state[5]
        x_force = (1-dt/tau) * self.state[4] + dt/tau * action[0] + np.random.normal(0., 0.01)# noise level similar to OFC
        y_force = (1-dt/tau) * self.state[5] + dt/tau * action[1] + np.random.normal(0., 0.01)
        
        # Apply forcefield:

        if (self.state[1] + y_vel*dt) < self.ff_top and (self.state[1] + y_vel*dt) > self.ff_bottom:
            x_vel = x_vel + self.ff_force[0]*dt
            y_vel = y_vel + self.ff_force[1]*dt


        
        # Update position and state
        self.pos = (x_pos, y_pos)
        self.state = np.array([x_pos, y_pos, x_vel, y_vel, x_force, y_force])
        
        # Check if finished 
        if self.goal_left <= self.pos[0] and self.goal_right >= self.pos[0]:         # reached goal in x dimension 
            if self.goal_top >= self.pos[1] and self.goal_bottom <= self.pos[1]: # reached goal in y dimension
                self.reward = 20 - np.linalg.norm(action, 2) * cost # INCREASE FOR SUCCESSFUL TRIALS
                self.done = True
                
        elif self.max_top <= self.pos[0] or self.max_bottom >= self.pos[0] or self.max_left >= self.pos[1] \
        or self.max_right <= self.pos[1]: # exited workspace
            self.reward = -10 - np.linalg.norm(action, 2) * cost
            self.done = True 
                
        elif self.time >= self.max_len:     # reached time limit
            self.reward = -10 - np.linalg.norm(action, 2) * cost
            self.done = True 
            
        else:                             # not finished 
            self.reward = 0 - np.linalg.norm(action, 2) * cost
            self.done = False 
        
        return self 
    
class MultiTarget(Workspace):
    pass
                           
def get_carried_action(old_action, action, act_weight = ACTION_WEIGHT):
        """Get new action based on action from previous timestep. 
        Carried action = (action_weight * old_action) + new_action 
        Params
        ======
        old_action: list length 2 [old_x_velocity, old_y_velocity]
        action: list length 2 [new_x_velocity, new_y_velocity]
        act_weight: weight applied to old velocity 
        """
        # NOT USED ANYMORE - CAN BE REMOVED
        new_x_velocity = old_action[0] * act_weight + action[0]
        new_y_velocity = old_action[1] * act_weight + action[1]
        
        return new_x_velocity, new_y_velocity                            
                           
        