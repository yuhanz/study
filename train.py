import numpy as np
import tensorflow as tf
import gym

import net_builder
import net_operation
import reinforcement

from matplotlib import animation
from JSAnimation.IPython_display import display_animation

import sys

# Create the environment and display the initial state
env = gym.make('LunarLander-v2')
observation_next = env.reset()

n_input = env.observation_space.shape[0]
n_output = env.action_space.n

OBSERVATION_INDEX = 0
TARGET_REWARD_INDEX = 2

MODEL_FILE_PATH = './model/model.ckpt'

[input, output, target, loss, train] = net_builder.build_net(n_input, n_output)


sess = tf.Session()
if len(sys.argv) > 1:
  net_operation.restore(sess, MODEL_FILE_PATH)
else:
  init = tf.global_variables_initializer()
  sess.run(init)

def step_and_collect_data(env, observation, sess, input, output):
  [action, evaluated_rewards] = reinforcement.choose_action(env, observation, sess, input, output)
  print '== action:' + str(action)
  print '== evaluated_rewards:', evaluated_rewards
  observation_next, reward, done, info = env.step(action)
  print '== actual_reward:', reward
  max_future_reward = net_operation.eval_and_max(sess, output, input, [observation_next]) if not done else -100
  target_rewards = reinforcement.to_target_reward(action, reward, max_future_reward, evaluated_rewards[0])
  return [target_rewards, observation_next, done, reward]

NUM_EPISODES = 300

reward_sums = []

for j in  range(1,NUM_EPISODES):
    records = []
    NUM_RECORDS = 300

    for i in range(1,NUM_RECORDS):
      observation = observation_next
      target_rewards, observation_next, done, actual_reward = step_and_collect_data(env, observation, sess, input, output)
      print "target_rewards:", target_rewards

      records.append([observation, target_rewards, actual_reward])
      if(done):
        print "=== done"
        break

    reward_sum = reduce(lambda s,r: s + r[2], records, 0)
    reward_sums.append(reward_sum)
    print "=== total actual reward:", reward_sum
    reinforcement.learn(sess, records, loss, train, input, target)
    if(NUM_RECORDS != len(records)):
      observation_next = env.reset()

print "=== average reward sum: ", reduce(lambda s,x: s + x, reward_sums, 0) / NUM_EPISODES

net_operation.save(sess, MODEL_FILE_PATH)
