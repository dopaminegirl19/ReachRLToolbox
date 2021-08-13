import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm


class Trajectories():
    
    def __init__(self, env):
        """"
        Params
        ======
        trajectories = list of trajectories, where each trajectory is an array or list of (x, y) coordinates at each timestep.
        env = environment in which trajectories were performed. indicates goal box, workspace dimensions, etc. 
        """
        
        self.trajectories = []
        self.max_len = env.max_len
        
        # goal box
        self.goal_left = env.goal_left
        self.goal_top = env.goal_top
        self.goal_right = env.goal_right
        self.goal_bottom = env.goal_bottom 
        
        # workspace dims
        self.max_top = env.max_top
        self.max_left = env.max_left
        self.max_bottom = env.max_bottom
        self.max_right = env.max_right
        
    def add_episode(self, positions):
        self.trajectories.append(positions)
        
    def plot(self, idx, color = 'magma', scale=True, boxcol = 'r', boxalpha = 0.1):
        """Plot a select number of indices.
        Params
        ======
        tr_id = int index of trajectory to plot
        cmap = colors to map with, from cm
        scale = if True, samples from the colormap based on length of trial. 
        """
        
        if scale:
            cmap = cm.get_cmap(color, len(self.trajectories[idx])) 
        else:
            cmap = cm.get_cmap(color, self.max_len)
            
        xlims = (self.max_left, self.max_right) 
        ylims = (self.max_bottom, self.max_top)
        
        fig, ax = plt.subplots()

        goal_patch = patches.Rectangle((self.goal_left, self.goal_bottom), self.goal_right-self.goal_left, \
                                       self.goal_top-self.goal_bottom, linewidth=1, alpha=boxalpha, \
                                       edgecolor=boxcol, facecolor=boxcol)
        
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        
        for i, pt in enumerate(self.trajectories[idx]):
            ax.plot(pt[0], pt[1], 'o', color=cmap.colors[i])

        ax.add_patch(goal_patch)
        
        return fig, ax
        
        
        
        