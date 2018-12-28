import numpy as np
import tensorflow as tf

import net_builder
import net_operation
import reinforcement
import gym_app


# Create the environment and display the initial state
env, observation_next, n_input, n_output = gym_app.loadGymEnv('LunarLander-v2')

MODEL_FILE_PATH = './model/model.ckpt'

[input, output, target, loss, train] = net_builder.build_net(n_input, n_output)

sess = gym_app.init_session(MODEL_FILE_PATH)
im = gym_app.initRender(env)

for j in range(1,10000):
  observation = observation_next
  [action, evaluated_rewards] = reinforcement.greedy_choose_action(env, observation, sess, input, output)
  observation_next, reward, done, info = env.step(action)
  print "=== action: ", action
  print "=== reward: ", reward
  gym_app.render(im, env)
  if done:
    print '=== done'
    print "=== total_steps:", j
    break
    #env.reset()
