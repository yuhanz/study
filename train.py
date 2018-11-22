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

[input, output, loss, train] = net_builder.build_net(n_input, n_output)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


records = []

for i in range(1,10):
  observation = observation_next
  [action, evaluated_rewards] = reinforcement.choose_action(env, observation, sess, input, output)
  print '== action:' + str(action)
  print '== evaluated_rewards:', evaluated_rewards
  observation_next, reward, done, info = env.step(action)
  max_future_reward = net_operation.eval_and_max(sess, output, input, [observation_next])
  target_rewards = reinforcement.to_target_reward(action, reward, max_future_reward, evaluated_rewards)
  records.append([observation, target_rewards])

for record in records:
  reinforcement.learn(sess, records, loss, train)
