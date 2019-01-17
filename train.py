import numpy as np
import tensorflow as tf
import gym_app

import net_builder
import net_operation
import reinforcement

from matplotlib import animation
from JSAnimation.IPython_display import display_animation

import sys

MODEL_FILE_PATH = './model/model.ckpt'
GYM_ENV_NAME = 'LunarLander-v2'

# Create the environment and display the initial state
env, observation_next, n_input, n_output = gym_app.loadGymEnv(GYM_ENV_NAME)
[input, output, target, loss, train] = net_builder.build_net(n_input, n_output, learning_rate = 0.001)
sess = gym_app.init_session(MODEL_FILE_PATH if len(sys.argv) > 1 else None)

NUM_EPISODES = 300

reward_sums = []

for j in  range(1,NUM_EPISODES):
    records = []
    NUM_RECORDS = 300

    positive_actual_reward = 0

    for i in range(1,NUM_RECORDS):
      observation = observation_next
      target_rewards, observation_next, done, actual_reward = gym_app.step_and_collect_data(env, observation, sess, input, output, lambda max_future_reward, reward, env: reward + positive_actual_reward + (20 if observation[6] and observation[7] else -15 if observation[6] or observation[7] else -10))
      print "target_rewards:", target_rewards

      positive_actual_reward += actual_reward if actual_reward > 0 else 0
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
