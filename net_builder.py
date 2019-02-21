import tensorflow as tf

def get_dimension(input):
  return input.shape.dims[1].value

def build_net(input_size, output_size, learning_rate = 0.01, hidden_layer_sizes = [32, 16]):
  with tf.variable_scope('eval'):
    l = input = build_input(input_size, 'input')
    for index, size in enumerate(hidden_layer_sizes):
      l = tf.nn.relu(build_layer(1+index, l, size))
    output = tf.nn.tanh(build_layer(len(hidden_layer_sizes)+1, l, output_size))
  with tf.variable_scope('loss'):
    target = build_target(output_size, 'output')
    loss = build_loss(output, target)
  with tf.variable_scope('train'):
    train = build_train(loss, learning_rate)
  return [input, output, target, loss, train]

def build_loss(output_layer, target):
  return tf.reduce_mean(tf.squared_difference(target, output_layer))

def build_train(loss, learning_rate):
  return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

def build_input(input_size, name):
  return tf.placeholder(tf.float32, [None, input_size], name=name)

def build_target(output_size, name):
  return tf.placeholder(tf.float32, [None, output_size], name=name)

def build_layer(level, input, layer_size):
  [w_init, b_init] = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
  ln = str(level)
  input_size = get_dimension(input)
  with tf.variable_scope('l'+ln):
      w = tf.get_variable('w'+ln, [input_size, layer_size], initializer=w_init);
      b = tf.get_variable('b'+ln, [1, layer_size], initializer=b_init);
      l = tf.matmul(input, w) + b
  return l

def build_layer_with_activation(level, input, layer_size):
  return tf.nn.relu(build_layer(level, input, layer_size))
