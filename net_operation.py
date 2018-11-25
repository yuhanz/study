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

def save(session, file_path):
  saver = tf.train.Saver()
  save_path = saver.save(session, file_path)
  print("Model saved in path: %s" % save_path)
  return save_path

def restore(session, file_path):
  saver = tf.train.Saver()
  saver.restore(session, file_path)
