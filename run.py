import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

import net_builder
import net_operation
import reinforcement


from matplotlib import animation
from JSAnimation.IPython_display import display_animation


# Create the environment and display the initial state
env = gym.make('CartPole-v0')
observation_next = env.reset()

n_input = 4
n_output = 2

OBSERVATION_INDEX = 0
TARGET_REWARD_INDEX = 2

MODEL_FILE_PATH = './model/model.ckpt'

[input, output, target, loss, train] = net_builder.build_net(n_input, n_output)

sess = tf.Session()
net_operation.restore(sess, MODEL_FILE_PATH)


firstframe = env.render(mode = 'rgb_array')
fig,ax = plt.subplots()
im = ax.imshow(firstframe)

for j in  range(1,30):
  observation = observation_next
  [action, evaluated_rewards] = reinforcement.greedy_choose_action(env, observation, sess, input, output)
  observation_next, reward, done, info = env.step(action)
  print "=== action: ", ['LEFT', 'RIGHT'][action]
  print "=== reward: ", reward
  frame = env.render(mode = 'rgb_array')
  im.set_data(frame)
  if done:
    print '=== done'
    env.reset()
