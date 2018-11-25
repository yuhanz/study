import numpy as np
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
init = tf.global_variables_initializer()
sess.run(init)

def step_and_collect_data(env, observation, sess, input, output):
  [action, evaluated_rewards] = reinforcement.choose_action(env, observation, sess, input, output)
  print '== action:' + str(action)
  print '== evaluated_rewards:', evaluated_rewards
  observation_next, reward, done, info = env.step(action)
  # if done:
  #   print '=== done'
  max_future_reward = net_operation.eval_and_max(sess, output, input, [observation_next])
  target_rewards = reinforcement.to_target_reward(action, reward, max_future_reward, evaluated_rewards[0])
  return [target_rewards, observation_next]


for j in  range(1,30):
    records = []

    for i in range(1,100):
      observation = observation_next
      target_rewards, observation_next = step_and_collect_data(env, observation, sess, input, output)
      records.append([observation, target_rewards])

    reinforcement.learn(sess, records, loss, train, input, target)

net_operation.save(sess, MODEL_FILE_PATH)
