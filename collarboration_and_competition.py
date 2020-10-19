#!/usr/bin/env python3

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

from unityagents import UnityEnvironment
import torch
from ddpg_agent import Agent

# Set to True if you want to (re)run the training or False to just watch the trained agent in action
train_now = False

# Initialize Unity environment
unity_env_path=os.path.join(os.getcwd(),'./Tennis_Linux/Tennis.x86_64')
env = UnityEnvironment(file_name=unity_env_path)

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Create an agent that controls the agents in the environment
agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)

# Train the agent with DDPG
if train_now:
    n_episodes=600
    average_len=100
    print_every=100
    scores_deque = deque(maxlen=average_len)
    scores = []
    solved = False
    score_average = 0
    score_max = 0
    # Train for n_episodes
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = 0
        steps = 0
        # Train until environment ends the episode
        while True:
            steps += 1
            # Let deep learning agent act based on states
            action_0 = agent.act(states[0])
            action_1 = agent.act(states[1])
            # Send action to Unity environment
            env_info = env.step([action_0,action_1])[brain_name]
            states_next = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # Save experiences to replay buffer
            agent.memorize(states[0], action_0, rewards[0], states_next[0], dones[0])
            agent.memorize(states[1], action_1, rewards[1], states_next[1], dones[1])
            # Learn
            agent.update_step()
            agent.update_step()
            states = states_next
            score += np.sum(rewards)/len(rewards)
            if np.any(dones):
                break
        # Check and track scores
        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)
        print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}'.format(i_episode, score_average, score), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, score_average))
            # Save coefficients to file if environment is solved with current network coefficients
            if score_average >= 0.1:
                if score_average > 0.5 and not solved:
                    solved = True
                    print('Environment solved after {} episodes.'.format(i_episode))
                if score >= score_max:
                    score_max = score
                    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#888888', linestyle='-', alpha=0.5)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig('training_scores.png')
    plt.show()

# Now watch the two trained agents in action
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
agent.reset()

for i in range(1, 3):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    steps = 0
    while True:
        steps += 1
        print('Step:',steps)
        action_0 = agent.act(states[0], add_noise=False)
        action_1 = agent.act(states[1], add_noise=False)
        env_info = env.step([action_0,action_1])[brain_name]
        states_next = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = states_next 
        if np.any(dones):
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

# Clean up
env.close()
