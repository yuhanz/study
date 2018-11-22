import tensorflow as tf
import numpy as np

import net_operation
import net_builder

OBSERVATION_INDEX = 0
TARGET_REWARD_INDEX = 1



def choose_action(env, observation, sess, input, output, epsilon = 0.9):
  evaluated = net_operation.eval(sess, output, input, np.array([observation]))
  if np.random.uniform() >= epsilon:
    action = np.random.randint(0, net_builder.get_dimension(output))
  else:
    action = np.argmax(evaluated)
  return [action, evaluated]

def to_target_reward(action, reward, max_future_reward, evaluated_reward, gamma = 0.9):
  target = evaluated_reward
  target[0][action] = reward + max_future_reward * gamma
  return target

def learn(sess, records, loss, train):
  inputs = map(lambda r : r[OBSERVATION_INDEX],records)
  targets = map(lambda r : r[TARGET_REWARD_INDEX],records)
  sess.run(train, feed_dict={'input.eval': inputs, 'loss.output': targets})
  loss_value = sess.run(loss, feed_dict={'input.eval': inputs, 'loss.output': targets})
  print "loss: " + str(loss_value)
