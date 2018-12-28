import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import net_operation
import reinforcement

def loadGymEnv(name):
# Create the environment and display the initial state
  env = gym.make(name)
  observation_next = env.reset()

  n_input = env.observation_space.shape[0]
  n_output = env.action_space.n
  return [env, observation_next, n_input, n_output]

def initRender(env):
  firstframe = env.render(mode = 'rgb_array')
  fig,ax = plt.subplots()
  im = ax.imshow(firstframe)
  return im

def render(im, env):
  frame = env.render(mode = 'rgb_array')
  im.set_data(frame)

def init_session(modelFilePath):
  sess = tf.Session()
  if modelFilePath:
    net_operation.restore(sess, modelFilePath)
  else:
    init = tf.global_variables_initializer()
    sess.run(init)
  return sess

def step_and_collect_data(env, observation, sess, input, output, doneRewardFn = None):
  [action, evaluated_rewards] = reinforcement.choose_action(env, observation, sess, input, output)
  print '== action:' + str(action)
  print '== evaluated_rewards:', evaluated_rewards
  observation_next, reward, done, info = env.step(action)
  print '== actual_reward:', reward
  max_future_reward = net_operation.eval_and_max(sess, output, input, [observation_next])
  if done and doneRewardFn:
    max_future_reward = doneRewardFn(max_future_reward, reward, env)
  target_rewards = reinforcement.to_target_reward(action, reward, max_future_reward, evaluated_rewards[0])
  return [target_rewards, observation_next, done, reward]
