[//]: # (Image References)

[tennis_environment]: tennis_environment.png "Tennis environment"

# Udacity deep reinforcement learning course - project 3

### Introduction

For this project, I trained an agent to play tennis against itself. That means
we consider a multi agent reinforcement learning problem. The tennis environment
can be visualized in a Unity application. The agent is based on the the actor
critic deep neural network architecture adapted to the multi agent problem. 

![Tennis environment][tennis_environment]

A reward of +0.1 is provided for each time the agent hits the ball over the net.
In the other cases, i.e, that the ball hits the ground or goes out of bounds
after being hit, the reward is -0.01.

Due to the continuous nature of the racket movement, ball position and opponent
position, the state space is very large. Basically just limited by the numeric
precision.

The agent can observe 8 values regarding position, velocity, of the racket and
the ball. It also gets the observations stacked over 3 consecutive time steps to
be able to better estimate the balls trajectory. Given this information, the
agent has to learn how to best select actions to move the racket. As the next
action it can decide how to move the racket on the 2D plane inside its half of
the court. Hence the size of the (vector) action space is 2.

The task is episodic as it ends with the ball hitting the ground or going out of
bounds. In order to solve the environment, the agent must get an average score
of +0.5 over 100 consecutive episodes after taking the maximum over both
instances of the agent in the game. This enforces a mixed competitive and
cooperative behavior.

### Unity environment download

1. Download the environment from one of the links below. You need only select
   the environment that matches your operating system:

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Extract it to a subdirectory.
3. Update the path to the executable in the Python script.

### Python dependencies

Please make sure the following Python requirements are fulfilled in your environment:

- jupyter
- unityagents
- numpy
- matplotlib
- torch

This can be done with

    pip3 install -r requirements.txt

Or your own favorite way of Python dependency management.

### How to run

After downloading and extracting the Unity environment, execute the
[collarboration_and_competition.py](collarboration_and_competition.py) Python
script in order to train the agent and/or see the trained agent in action.

### Solution report

See [report](REPORT.md) for more information about my solution.
