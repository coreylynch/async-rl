import tensorflow as tf

ACTIONS = 4

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def weight_variable(name, shape):
    with tf.device('/cpu:0'):
      initial = tf.truncated_normal(shape, stddev = 0.01)
      var = tf.Variable(initial)
    return var

def bias_variable(name, shape):
    with tf.device('/cpu:0'):
      initial = tf.constant(0.01, shape = shape)
      var = tf.Variable(initial)
    return var

def build_network(state):
  """Build the Deep Q-Network

  Args:
    state_placeholder: placeholder for current game state (previous agent_history_length frames)

  Returns:
    Q(s,a) values
  """

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = weight_variable('weights', shape=[8, 8, 4, 32])
    conv = tf.nn.conv2d(state, kernel, strides=[1, 4, 4, 1], padding='SAME')
    biases = bias_variable('biases', shape=[32])
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = weight_variable('weights', shape=[4, 4, 32, 64])
    conv = tf.nn.conv2d(conv1, kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = bias_variable('biases', shape=[64])
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = weight_variable('weights', shape=[3, 3, 64, 64])
    conv = tf.nn.conv2d(conv2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = bias_variable('biases', shape=[64])
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)

  # fc1
  with tf.variable_scope('fc1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in conv3.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(conv3, [1, dim])

    weights = weight_variable('weights', shape=[dim, 512])
    biases = bias_variable('biases', shape=[512])
    fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # out
  with tf.variable_scope('out') as scope:
    weights = weight_variable('weights', shape=[512, ACTIONS])
    biases = bias_variable('biases', shape=[ACTIONS])
    q_values = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)

  return q_values

def loss(q_values, a, y):
  """Build the DQN Loss

  Args:
    q_values: op that outputs the q_values of the DQN, given some environment state
    a: One hot vector representing chosen action at time t
    y: placeholder op that outputs the target 
  Returns:
    MSE loss
  """
  action_value = tf.reduce_sum(tf.mul(q_values, a), reduction_indices = 1)
  cost = tf.reduce_mean(tf.square(y - action_value))
  return cost
