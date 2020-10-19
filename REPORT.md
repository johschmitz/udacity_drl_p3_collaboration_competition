[//]: # (Image References)

[training_scores]: training_scores.png "Training_scores"

# Project 3: Mixed cooperative/competitive tennis playing - report

This report describes my solution and some findings along the way.

### Solution architecture and strategy

My solution uses an actor critic deep neural network architecture with a deep
deterministic policy gradient (DDPG) training approach. It is based on the
Udacity DDPG implementation used to solve the pendulum environment. I also
evaluated the multi agent DDPG algorithm described in  the 2017 (revised 2020)
paper "[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf])"
by Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel and Igor Mordatch.
However, the problem didn't seem to be suited very well to be solved with MADDPG
due to the low number of agents and the relatively weak effect on each other.
When the ball crosses the net it is either so fast that it goes out, otherwise
it is almost always possible for the player to get it. Hence the room for
improvement with MADDPG in comparison appears to be low. Furthermore, the tennis
environment suffers from a severe tendency for instable convergence and my
MADDPG implementations where not able to converge to any good policies. Hence, I
am describing my DDPG solution here which performed reasonably well.

In order to successfully train the actor critic network with DDPG to perform
well on the tennis game, the following adoptions had to be implemented

- The actor network is used twice in a serial pattern to act for each of the
  agents in the Unity environment.
- The Ornstein-Uhlenbeck noise process has been replaced with an aditive normal
  distributed noise.
- The noise process is now decaying over time.
- The noise process is now normal distributed.
- The variance and decay of the noise and hence the exploration/eploitation
  trade-off have been carefully tweaked until the algorithm converged every time.

The network architectures of the submitted actor and critic networks is as follows:

Actor network
- Layer1: Linear + BatchNorm1d, input width:24, output width: 256, activation: ReLu
- Layer2: Linear, input width: 256, output width: 128, activation: ReLu
- Layer3: Linear, input width: 128, output width: 4, activation: tanh

Critic network:
- Layer1: Linear + BatchNorm1d, input width: 24, output width: 256, activation: ReLu
- Layer2: Linear, input width: 256 + 2, output width: 128, activation: ReLu
- Layer3: Linear, input width: 128 output width: 1, activation: tanh

With this I got a successful training result as seen in the plot below with the
environment being solved after 500 Episodes (average score of last 100 episodes
above 0.5). Overall a maximum score of about 2.6 was reached, which means the
ball would not drop over a full episode.

![Training scores][training_scores]

The following parameters have been used for the submitted result:

- Replay buffer size: <img src="https://render.githubusercontent.com/render/math?math=1\times10^{-5}">
- Batch size: <img src="https://render.githubusercontent.com/render/math?math=128">
- Discount factor: <img src="https://render.githubusercontent.com/render/math?math=\gamma=0.99">
- Soft update coefficient: <img src="https://render.githubusercontent.com/render/math?math=\tau=1\times10^{-3}">
- Learning rate of actor: <img src="https://render.githubusercontent.com/render/math?math=1\times10^{-4}">
- Learning rate of critic: <img src="https://render.githubusercontent.com/render/math?math=1\times10^{-3}">
- Standard deviation of noise: <img src="https://render.githubusercontent.com/render/math?math=\sigma=0.25">
- Weight decay of Adam optimizer: <img src="https://render.githubusercontent.com/render/math?math=0">
- Exploration decay coefficient: <img src="https://render.githubusercontent.com/render/math?math=\epsilon_\text{decay}=0.99991">
- Minimum exploration decay coefficient: <img src="https://render.githubusercontent.com/render/math?math=\epsilon_\text{min}=0.1">

### Findings

- It was very difficult to get to a point where the training was actually
  converging. Until then the model weights where always diverging and the agent
  was always running towards the net or the back of the court.
- The most important things to achieve a converging training was to tweak the
  variance noise process and even more important the decay of the noise over time.

### Things to look at

In the future I might have a look at

- Making MADDPG work.
- Creating a network with a memory that accepts more than one input frame or use
  a recurrent network. This could possibly even better estimate the ball
  trajectory and play in more challenging tennis environments.
- Invest much more time in studying best practices that help to make the training
  converging and stable.