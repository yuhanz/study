import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display

import gym

from matplotlib import animation


# Create the environment and display the initial state
env = gym.make('Pendulum-v0')
observation = env.reset()
firstframe = env.render(mode = 'rgb_array')
fig,ax = plt.subplots()
im = ax.imshow(firstframe)


# Function that defines what happens when you click one of the buttons
frames = []
def onclick(action):
    global frames
    observation, reward, done, info = env.step(action)
    frame = env.render(mode = 'rgb_array')
    im.set_data(frame)
    frames.append(frame)
    #if done:
    #    env.reset()


# Show the buttons to control the cart
[onclick([1]) for i in range(1,5)]
[onclick([-1]) for i in range(1,5)]
[onclick([1]) for i in range(1,5)]
[onclick([-1]) for i in range(1,5)]
[onclick([1]) for i in range(1,5)]
[onclick([-1]) for i in range(1,10)]
[onclick([1]) for i in range(1,10)]
[onclick([-1]) for i in range(1,10)]
onclick([-1])
onclick([-1])
onclick([-1])
onclick([-1])
onclick([-1])
onclick([-1])
onclick([1])
onclick([1])
onclick([1])
onclick([1])
onclick([1])
onclick([1])
onclick([-1])
onclick([-1])
onclick([-1])
onclick([-1])
onclick([-1])
onclick([-1])
