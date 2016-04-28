import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense

GAMMA = 0.99

def build_network(state, namespace, num_actions):
  with tf.device("/cpu:0"):
    with tf.variable_scope(namespace) as scope:
      model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), init='he_uniform', activation='relu', border_mode='same')(state)
      model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), init='he_uniform', activation='relu', border_mode='same')(model)
      model = Flatten()(model)
      model = Dense(output_dim=256, init='he_uniform', activation='relu')(model)
      q_values = Dense(output_dim=num_actions, init='he_uniform', activation='linear')(model)
  return q_values

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
  # Q value of action taken in state s
  action_value = tf.reduce_sum(tf.mul(q_values, action), reduction_indices=1)
  
  # Compute y
  # if terminal:
  #   y = r
  # else:
  #   y = r + gamma * max target_q_s_a
  y = reward + (1-terminal) * GAMMA * tf.reduce_max(target_q_values, reduction_indices=[1])
  
  cost = tf.reduce_mean(tf.square(y - action_value))
  # cost = tf.reduce_sum(tf.square(y - action_value)) # TRY THIS NEXT

  return cost
