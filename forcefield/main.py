from ff_environment import Workspace
from agent import Agent
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches



env = Workspace()
env_info = env.reset()

# size of each action
action_size = env.action_size
print('Size of each action:', action_size)

# examine the state space
state = env_info.state
state_size = len(state)
print('The agent observes a state with length: {}'.format(state_size))
print('The starting state looks like:', state)

# Instantiate the agent:
agent = Agent(state_size, action_size, random_seed=2)


# train the agent with ddpg
def ddpg(n_episodes=3000, max_t=1000, print_every=1000):
    scores = []
    trajectories = []
    actions_tracker = []
    scores_deque = deque(maxlen=print_every)

    for i_episode in range(n_episodes):
        env_info = env.reset()
        state = env_info.state  # current state
        score = 0  # initialize agent scores
        trajectory = [state[:2]]  # initialize trajectory
        actions = [state[2:]]
        agent.reset()  # reset noise process for action exploration

        for t in range(max_t):

            action = agent.act(state)

            env_info = env.step(action)  # send action to environment
            next_state = env_info.state  # get next state
            reward = env_info.reward  # get reward
            done = env_info.done  # see if trial is finished

            agent.step(state, action, reward, next_state, done)

            score += reward  # update the score (for each agent)
            state = next_state  # enter next states
            trajectory.append(env_info.pos)
            actions.append(action)

            if done:
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        trajectories.append(trajectory)
        actions_tracker.append(actions)

        print('\rEpisode {} \tAverage Reward: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            torch.save(agent.actor_local.state_dict(), 'actor_model.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_model.pth')
            print('\rEpisode {} \tAverage Reward: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 0:
            print('\nEnvironment solved in {:d} episodes!\t Average Score: {:.2f}'.format(i_episode,
                                                                                          np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'actor_solved.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_solved.pth')
            break

    return scores, trajectories, actions_tracker

#plotting the value of reward/episode (left) and the converged (or last) trajectory (right)
scores, trajectories, actions_tracker = ddpg()
start_pos = (.5, 1)
padding=5
xlims = (start_pos[0] - padding, start_pos[0] + padding)
ylims = (start_pos[1] - padding, start_pos[1] + padding)
goal_top = 0
goal_left = .1
goal_right = .9
goal_bottom = -.8
start_pos = (.5, 1)
goal_patch = patches.Rectangle((goal_left, goal_bottom), goal_right-goal_left, goal_top-goal_bottom,
                               linewidth=1, alpha=0.1, edgecolor='r', facecolor='r')
fig, axs = plt.subplots(1,2)
axs[0].plot(range(len(scores)),scores)

axs[1].plot(start_pos[0], start_pos[1], 'x',color='b')
count = 0
axs[1].plot(trajectories[-1][0][0],trajectories[-1][0][1],'+',color='g',markersize=30)
for pt in trajectories[-1]:
    axs[1].plot(pt[0],pt[1],'o',color='r')
    count+=1
print(count)
axs[1].set_xlim(xlims[0], xlims[1])
axs[1].set_ylim(ylims[0], ylims[1])
axs[1].add_patch(goal_patch)
plt.show()


# Representing the different state variables (positions and velocities)
time_vec = np.zeros(count)
state_vec = np.zeros((state_size,count))
env_model = env.reset()
init_state=env_model.state
score = 0
state_vec[:,0] = init_state
for t in range(count-1):
    action = agent.act(state_vec[:,t])

    env_info = env_model.step(action)  # send action to environment
    next_state = env_info.state  # get next state
    reward = env_info.reward  # get reward
    done = env_info.done  # see if trial is finished
    score += reward  # update the score (for each agent)
    state_vec[:,t+1] = next_state  # enter next states

    if done:
        break

fig, ax=plt.subplots(2,2)
ax[0,0].plot(range(count),state_vec[0,:],'r')
ax[0,1].plot(range(count),state_vec[1,:],'r')
ax[1,0].plot(range(count),state_vec[2,:],'r')
ax[1,1].plot(range(count),state_vec[3,:],'r')
plt.show()
