import tensorflow as tf

def get_dimension(input):
  return input.shape.dims[1].value

def build_net(input_size, output_size, learning_rate = 0.01):
  layer1_size = 32
  layer2_size = 16

  with tf.variable_scope('eval'):
    input = build_input(input_size, 'input')
    l1 = build_layer(1, input, layer1_size)
    l2 = build_layer(2, l1, layer2_size)
    output = build_layer(3, l2, output_size)
  with tf.variable_scope('loss'):
    target = build_target(output_size, 'output')
    loss = build_loss(output, target)
  with tf.variable_scope('train'):
    train = build_train(loss, learning_rate)
  return [input, output, target, loss, train]

def build_loss(output_layer, target):
  return tf.reduce_mean(tf.squared_difference(output_layer, target))

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
      l = tf.nn.relu(tf.matmul(input, w) + b)
  return l
