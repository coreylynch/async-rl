import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense

ACTIONS = 4
GAMMA = 0.99

def build_network(state, namespace):
  with tf.device("/cpu:0"):
    with tf.variable_scope(namespace) as scope:
      model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), init='he_uniform', activation='relu', border_mode='same')(state)
      model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), init='he_uniform', activation='relu', border_mode='same')(model)
      model = Flatten()(model)
      model = Dense(output_dim=256, init='he_uniform', activation='relu')(model)
      q_values = Dense(output_dim=ACTIONS, init='he_uniform', activation='linear')(model)
  return q_values

# def _variable_on_cpu(name, shape, initializer):
#   """Helper to create a Variable stored on CPU memory.

#   Args:
#     name: name of the variable
#     shape: list of ints
#     initializer: initializer for Variable

#   Returns:
#     Variable Tensor
#   """
#   with tf.device('/cpu:0'):
#     var = tf.get_variable(name, shape, initializer=initializer)
#   return var

# def weight_variable(shape):
#     with tf.device('/cpu:0'):
#       initial = tf.truncated_normal(shape, stddev = 0.01)
#       var = tf.Variable(initial)
#     return var

# def bias_variable(shape):
#     with tf.device('/cpu:0'):
#       initial = tf.constant(0.1, shape = shape)
#       var = tf.Variable(initial)
#     return var

# def conv2d(x, W, stride):
#     return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

# def build_target_network_concrete(state):
#     """
#     Following section 5.1
#     """
#     W_conv1 = weight_variable([8, 8, 4, 16])
#     b_conv1 = bias_variable([16])

#     W_conv2 = weight_variable([4, 4, 16, 32])
#     b_conv2 = bias_variable([32])
    
#     W_fc1 = weight_variable([3872, 256])
#     b_fc1 = bias_variable([256])

#     W_fc2 = weight_variable([256, ACTIONS])
#     b_fc2 = bias_variable([ACTIONS])

#     # hidden layers
#     h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1)

#     h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#     h_conv2_flat = tf.reshape(h_conv2, [-1, 3872])

#     h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

#     # readout layer
#     q_values = tf.matmul(h_fc1, W_fc2) + b_fc2

#     return q_values, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

# def build_network_concrete(state):
#     """
#     Following section 5.1
#     """
#     W_conv1 = weight_variable([8, 8, 4, 16])
#     b_conv1 = bias_variable([16])

#     W_conv2 = weight_variable([4, 4, 16, 32])
#     b_conv2 = bias_variable([32])
    
#     W_fc1 = weight_variable([3872, 256])
#     b_fc1 = bias_variable([256])

#     W_fc2 = weight_variable([256, ACTIONS])
#     b_fc2 = bias_variable([ACTIONS])

#     # hidden layers
#     h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1)

#     h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#     h_conv2_flat = tf.reshape(h_conv2, [-1, 3872])

#     h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

#     # readout layer
#     q_values = tf.matmul(h_fc1, W_fc2) + b_fc2

#     return q_values, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]


# def build_network3(state):
#     """
#     Following section 5.1
#     """
#     W_conv1 = weight_variable([8, 8, 4, 16])
#     b_conv1 = bias_variable([16])

#     W_conv2 = weight_variable([4, 4, 16, 32])
#     b_conv2 = bias_variable([32])
    
#     W_fc1 = weight_variable([3872, 256])
#     b_fc1 = bias_variable([256])

#     W_fc2 = weight_variable([256, ACTIONS])
#     b_fc2 = bias_variable([ACTIONS])

#     # hidden layers
#     h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1)

#     h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#     h_conv2_flat = tf.reshape(h_conv2, [-1, 3872])

#     h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

#     # readout layer
#     q_values = tf.matmul(h_fc1, W_fc2) + b_fc2

#     return q_values

# def build_network2(state):
#     W_conv1 = weight_variable([8, 8, 4, 32])
#     b_conv1 = bias_variable([32])

#     W_conv2 = weight_variable([4, 4, 32, 64])
#     b_conv2 = bias_variable([64])

#     W_conv3 = weight_variable([3, 3, 64, 64])
#     b_conv3 = bias_variable([64])
    
#     W_fc1 = weight_variable([7744, 512])
#     b_fc1 = bias_variable([512])

#     W_fc2 = weight_variable([512, ACTIONS])
#     b_fc2 = bias_variable([ACTIONS])

#     # hidden layers
#     h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1)

#     h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#     h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

#     h_conv3_flat = tf.reshape(h_conv3, [-1, 7744])

#     h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

#     # readout layer
#     q_values = tf.matmul(h_fc1, W_fc2) + b_fc2

#     return q_values

# def build_network(state):
#   """Build the Deep Q-Network

#   Args:
#     state_placeholder: placeholder for current game state (previous agent_history_length frames)

#   Returns:
#     Q(s,a) values
#   """

#   # conv1
#   with tf.variable_scope('conv1') as scope:
#     kernel = weight_variable([8, 8, 4, 32])
#     conv = tf.nn.conv2d(state, kernel, strides=[1, 4, 4, 1], padding='SAME')
#     biases = bias_variable([32])
#     bias = tf.nn.bias_add(conv, biases)
#     conv1 = tf.nn.relu(bias, name=scope.name)

#   # conv2
#   with tf.variable_scope('conv2') as scope:
#     kernel = weight_variable([4, 4, 32, 64])
#     conv = tf.nn.conv2d(conv1, kernel, strides=[1, 2, 2, 1], padding='SAME')
#     biases = bias_variable([64])
#     bias = tf.nn.bias_add(conv, biases)
#     conv2 = tf.nn.relu(bias, name=scope.name)

#   # conv3
#   with tf.variable_scope('conv3') as scope:
#     kernel = weight_variable([3, 3, 64, 64])
#     conv = tf.nn.conv2d(conv2, kernel, strides=[1, 1, 1, 1], padding='SAME')
#     biases = bias_variable([64])
#     bias = tf.nn.bias_add(conv, biases)
#     conv3 = tf.nn.relu(bias, name=scope.name)

#   # fc1
#   with tf.variable_scope('fc1') as scope:
#     # Move everything into depth so we can perform a single matrix multiply.
#     dim = 1
#     for d in conv3.get_shape()[1:].as_list():
#       dim *= d
#     reshape = tf.reshape(conv3, [1, dim])

#     weights = weight_variable([dim, 512])
#     biases = bias_variable([512])
#     fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

#   # out
#   with tf.variable_scope('out') as scope:
#     weights = weight_variable([512, ACTIONS])
#     biases = bias_variable([ACTIONS])
#     q_values = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)

#   return q_values

def loss(q_values, target_q_values, action, reward, terminal):
  """Build the one-step DQN Loss

  Args:
    q_values: op that outputs the q_values of the DQN, given some environment state
    target_q_values: op that outputs the target_q_values of the DQN, given some environment state
    action: One hot vector representing chosen action at time t
    reward: instantaneous reward from executing action in state
    terminal: 1 if new_state is terminal, 0 otherwise
  Returns:
    MSE loss
  """
  CLIP_DELTA = 0 # not sure about this implementation yet

  # Q value of action taken in state s
  action_value = tf.reduce_sum(tf.mul(q_values, action), reduction_indices=1)
  
  # Compute y
  # if terminal:
  #   y = r
  # else:
  #   y = r + gamma * max target_q_s_a
  y = reward + (1-terminal) * GAMMA * tf.reduce_max(target_q_values, reduction_indices=[1])
  
  cost = tf.reduce_mean(tf.square(y - action_value))

  return cost
