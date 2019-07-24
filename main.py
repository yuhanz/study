import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display

import gym
from matplotlib import animation

import control


# Create the environment and display the initial state
env = gym.make('Pendulum-v0')
observation = env.reset()
firstframe = env.render(mode = 'rgb_array')
fig,ax = plt.subplots()
im = ax.imshow(firstframe)

x_desired = [1.0, 0.0, 0.0]
# when the pendulum is at up-right:
# theta = 0; thetaDot = 0
# x = [cos(theta); sin(theta); theta']
# x.desired = [1, 0, 0]
# u = -K * (X - X.desired) = -K* (X - [1,0,0])
# X' = [-sin(theta); cos(theta); theta"] = Ax + Bu
# theta" = (m*l**2) * u - g*sin(theta)
# m = 1; l = 1; g = 10
# theta" = [u - 10 sin(theta)]
# X' = [cos'(theta); sin'(theta); theta"]
# X' = [-sin(theta); cos(theta); u - 10 sin(theta)]
# X' = [0* cos(theta) -1 * sin(theta) + 0u;
#       1* cos(theta) +0 * sin(theta) + 0u;
#       0* cos(theta) -10* sin(theta) + 1u]
# X' = Ax + Bu
# A = [0, -1, 0; 1, 0, 0; 0, -10, 1]
# B = [0,0,1]
# Make up Q and R
# Q = [1, 0, 0; 0, 1, 0; 0, 0, 10]
# R = [0.01]

# -- second attempt:
# theta" = 3* (m*l**2) * u - 3 * g/ (2*l) *sin(theta + pi)
# theta" = 3u + 15 sin(theta)
# X' = [0* cos(theta) -1 * sin(theta) + 0u;
#       1* cos(theta) +0 * sin(theta) + 0u;
#       0* cos(theta) +15* sin(theta) + 3u]
#


#A = np.matrix('0.0, -1.0, 0.0; 1.0, 0.0, 0.0; 0.0, -10.0, 1.0')
A = np.matrix('0.0, -1.0, 0.0; 1.0, 0.0, 0.0; 0.0, 15.0, 3.0')
B = np.matrix('0.0; 0.0; 1.0')
Q = np.matrix('1.0, 0.0, 0.0; 0.0, 1.0, 0.0; 0.0, 0.0, 3.0')
R = np.matrix('0.1')

K, S, E = control.lqr(A, B, Q, R)

print("K:", K)
# import pdb; pdb.set_trace()

def stepByLQR(observation, env):
    X = observation
    # u = -K * (X - X.desired) = -K* (X - [1,0,0])
    u = np.matmul(-K, X - [1,0,0])
    print("u:", u)
    return env.step([u])

# Function that defines what happens when you click one of the buttons
frames = []
def onclick(action):
    global frames
    observation, reward, done, info = env.step(action)
    print(observation)
    frame = env.render(mode = 'rgb_array')
    im.set_data(frame)
    frames.append(frame)
    #if done:
    #    env.reset()

def guided_run(observation):
    global frames
    print("Observation0:", observation)
    observation, reward, done, info = stepByLQR(observation, env)
    print("Observation: ", observation)
    frame = env.render(mode = 'rgb_array')
    im.set_data(frame)
    frames.append(frame)
    return observation[0]

for i in range(1,500):
    observation = guided_run(observation)



# Show the buttons to control the cart
# [onclick([0]) for i in range(1,50)]
# [onclick([1]) for i in range(1,5)]
# [onclick([-1]) for i in range(1,5)]
# [onclick([1]) for i in range(1,5)]
# [onclick([-1]) for i in range(1,5)]
# [onclick([1]) for i in range(1,5)]
# [onclick([-1]) for i in range(1,10)]
# [onclick([1]) for i in range(1,10)]
# [onclick([-1]) for i in range(1,10)]
# onclick([-1])
# onclick([-1])
# onclick([-1])
# onclick([-1])
# onclick([-1])
# onclick([-1])
# onclick([1])
# onclick([1])
# onclick([1])
# onclick([1])
# onclick([1])
# onclick([1])
# onclick([-1])
# onclick([-1])
# onclick([-1])
# onclick([-1])
# onclick([-1])
# onclick([-1])
