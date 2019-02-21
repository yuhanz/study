import numpy as np
import tensorflow as tf
import gym_app

import net_builder
import net_operation
import reinforcement

import json_util

from matplotlib import animation
from JSAnimation.IPython_display import display_animation

import sys
import mlflow

GYM_ENV_NAME = 'LunarLander-v2'
MODEL_FILE_PATH_MAP = { \
    'LunarLander-v2': './model/model.ckpt', \
    'CartPole-v0': './models/car-pole-model/model.ckpt' \
}

MODEL_FILE_PATH = MODEL_FILE_PATH_MAP[GYM_ENV_NAME]
LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZES = [32, 16, 8]
EPSILON = 0.8

print('GYM_ENV_NAME: %s', GYM_ENV_NAME)
print('MODEL_FILE_PATH: %s', MODEL_FILE_PATH)

# Create the environment and display the initial state
env, observation_next, n_input, n_output = gym_app.loadGymEnv(GYM_ENV_NAME)
[input, output, target, loss, train] = net_builder.build_net(n_input, n_output, learning_rate = LEARNING_RATE, hidden_layer_sizes = HIDDEN_LAYER_SIZES)

RESUME_TRAINING = len(sys.argv) > 1
sess = gym_app.init_session(MODEL_FILE_PATH if RESUME_TRAINING else None)

NUM_EPISODES = 300
NUM_RECORDS = 600

mlflow.log_param('GYM_ENV_NAME', GYM_ENV_NAME)
mlflow.log_param('MODEL_FILE_PATH', MODEL_FILE_PATH)
mlflow.log_param('LEARNING_RATE', LEARNING_RATE)
mlflow.log_param('RESUME_TRAINING', RESUME_TRAINING)
mlflow.log_param('NUM_EPISODES', NUM_EPISODES)
mlflow.log_param('NUM_RECORDS', NUM_RECORDS)
mlflow.log_param('HIDDEN_LAYER_SIZES', HIDDEN_LAYER_SIZES)
mlflow.log_param('EPSILON', EPSILON)


import datetime
start_time = datetime.datetime.now()

reward_sums = []

for j in  range(1,NUM_EPISODES):
    records = []

    positive_actual_reward = 0
    negatives_actual_reward = 0
    reward_streak = 0    # how many negative reward appeared in a row? positive value if positive rewards in a row; negative value if negative rewards in a row

    wheel_touchdown_reward = 0

    for i in range(1,NUM_RECORDS):
      observation = observation_next

      num_wheels_touchdown = observation[6] + observation[7];
      wheel_touchdown_reward = 50 if num_wheels_touchdown >= 2 else (num_wheels_touchdown-2) * 50

      target_rewards, observation_next, done, actual_reward = gym_app.step_and_collect_data(env, observation, sess, input, output, lambda max_future_reward, reward, env: reward + positive_actual_reward + negatives_actual_reward + wheel_touchdown_reward, epsilon = EPSILON)
      print "target_rewards:", target_rewards

      if actual_reward > 0:
        reward_streak = 1 if reward_streak < 0 else reward_streak + 1
      if actual_reward < 0:
        reward_streak = -1 if reward_streak > 0 else reward_streak - 1

      positive_actual_reward += actual_reward if actual_reward > 0 else 0
      negatives_actual_reward += actual_reward if actual_reward < -4 else 0    # punish large negative values
      negatives_actual_reward += actual_reward if reward_streak < -3 else 0     # punish long period of negative values

      records.append([observation, target_rewards, actual_reward])
      if(done):
        print "=== done"
        break


    reward_sum = reduce(lambda s,r: s + r[2], records, 0)
    reward_sums.append(reward_sum)
    print "=== total actual reward:", reward_sum
    mlflow.log_metric('sum_actual_reward', reward_sum)
    reinforcement.learn(sess, records, loss, train, input, target)
    if records[-1][2] > 0:
        print "Succeeded!"
        mlflow.log_metric('success', 1)
        json_util.save_to_file(records, '/tmp/success_records_{}.json'.format(j));

    if(NUM_RECORDS != len(records)):
      observation_next = env.reset()

print "=== average reward sum: ", reduce(lambda s,x: s + x, reward_sums, 0) / NUM_EPISODES
print "total_time: " + str(datetime.datetime.now() - start_time)

net_operation.save(sess, MODEL_FILE_PATH)
mlflow.log_artifacts(MODEL_FILE_PATH[0:MODEL_FILE_PATH.rfind('/')])
