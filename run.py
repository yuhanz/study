import sys
import numpy as np
import tensorflow as tf

import net_builder
import net_operation
import reinforcement
import gym_app

import mlflow

gym_app_name = (sys.argv + ['LunarLander-v2'])[1]

print('Running %s', gym_app_name)

MODEL_FILE_PATH_MAP = { \
    'LunarLander-v2': './model/model.ckpt', \
    'CartPole-v0': './models/car-pole-model/model.ckpt' \
}
HIDDEN_LAYER_SIZES = [32, 16, 8]

# Create the environment and display the initial state
env, observation_next, n_input, n_output = gym_app.loadGymEnv(gym_app_name)

MODEL_FILE_PATH = MODEL_FILE_PATH_MAP[gym_app_name]

[input, output, target, loss, train] = net_builder.build_net(n_input, n_output, hidden_layer_sizes = HIDDEN_LAYER_SIZES)

sess = gym_app.init_session(MODEL_FILE_PATH)
im = gym_app.initRender(env)

total_reward = 0

for j in range(1,10000):
  observation = observation_next
  [action, evaluated_rewards] = reinforcement.greedy_choose_action(env, observation, sess, input, output)
  observation_next, reward, done, info = env.step(action)
  print "=== action: ", action
  print "=== reward: ", reward

  mlflow.log_metric('reward', reward)
  total_reward += reward

  gym_app.render(im, env)
  if done:
    print '=== done'
    print "=== total_steps:", j
    print "=== total_reward:", total_reward
    break
    #env.reset()
