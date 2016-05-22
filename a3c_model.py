import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense

def build_policy_and_value_networks(num_actions, agent_history_length, resized_width, resized_height):
    with tf.device("/cpu:0"):
        with tf.variable_scope("shared") as scope:
            state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
            model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', init='glorot_uniform', border_mode='same')(state)
            model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', init='glorot_uniform', border_mode='same')(model)
            model = Flatten()(model)
            model = Dense(output_dim=256, activation='relu', init='glorot_uniform')(model)
        with tf.variable_scope("policy") as scope:
            policy_network = Dense(output_dim=num_actions, activation='softmax', init='glorot_uniform')(model)
        with tf.variable_scope("value") as scope:
        	value_network = Dense(output_dim=1, activation='linear', init='glorot_uniform')(model)
    return state, policy_network, value_network