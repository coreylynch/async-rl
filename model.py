import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense

def build_network(namespace, num_actions, agent_history_length, resized_width, resized_height):
  with tf.device("/cpu:0"):
      with tf.variable_scope(namespace) as scope:
          state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
          model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(state)
          model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model)
          model = Flatten()(model)
          model = Dense(output_dim=256, activation='relu')(model)
          q_values = Dense(output_dim=num_actions, activation='linear')(model)
  return state, q_values