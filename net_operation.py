import tensorflow as tf
import numpy as np

def eval(session, output_layer, input, input_value):
  return session.run(output_layer, {input: input_value})

def eval_and_argmax(session, output_layer, input, input_value):
  values = eval(session, output_layer, input, input_value)
  return np.argmax(values)

def eval_and_max(session, output_layer, input, input_value):
  values = eval(session, output_layer, input, input_value)
  return np.argmax(values)


def learn(session, q_next, q_eval):
  session.run([q_next, q_eval], feed_dict={
    # ?
  })
