import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym_app

import net_builder
import net_operation
import reinforcement


from matplotlib import animation
from JSAnimation.IPython_display import display_animation


# Create the environment and display the initial state
env, observation_next, n_input, n_output = gym_app.loadGymEnv('CartPole-v0')

OBSERVATION_INDEX = 0
TARGET_REWARD_INDEX = 2

MODEL_FILE_PATH = './models/car-pole-model/model.ckpt'

[input, output, target, loss, train] = net_builder.build_net(n_input, n_output)
sess = gym_app.init_session(MODEL_FILE_PATH)
im = gym_app.initRender(env)

for j in  range(1,10000):
  observation = observation_next
  [action, evaluated_rewards] = reinforcement.greedy_choose_action(env, observation, sess, input, output)
  observation_next, reward, done, info = env.step(action)
  print "=== action: ", ['LEFT', 'RIGHT'][action]
  print "=== reward: ", reward
  gym_app.render(im, env)
  if done:
    print '=== done'
    print "=== total_steps:", j
    break
    #env.reset()
