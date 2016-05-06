#!/usr/bin/env python
import threading
import tensorflow as tf
import sys
import random
import numpy as np
import time
import gym
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque

from skimage.transform import resize
from skimage.color import rgb2gray

# Path params
EXPERIMENT_NAME = "pong_simple"
SUMMARY_SAVE_PATH = "/Users/coreylynch/dev/async-rl/summaries/"+EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = "/tmp/"+EXPERIMENT_NAME+".ckpt"
CHECKPOINT_INTERVAL=1

# Experiment params
GAME = "Pong-v0"
ACTIONS = 6
NUM_CONCURRENT = 4
NUM_EPISODES = 20000

AGENT_HISTORY_LENGTH = 4
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84

# Async params
NETWORK_UPDATE_FREQUENCY = 32
TARGET_NETWORK_UPDATE_FREQUENCY = 10000

# DQN Params
GAMMA = 0.99
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

# Epsilon params
INITIAL_EPSILON = 1.0
# FINAL_EXPLORATION_FRAME = 4000000
FINAL_EXPLORATION_FRAME = 1000000
EXPLORE = FINAL_EXPLORATION_FRAME

#Shared global parameters
T = 0

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # # input layer
    s = tf.placeholder("float", [None, 84, 84, 4])

    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
 
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
 
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([7744, 512])
    b_fc1 = bias_variable([512])
 
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
 
    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
 
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
 
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
 
    h_conv3_flat = tf.reshape(h_conv3, [-1, 7744])
 
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
 
    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
 
    return s, readout, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2

def copyTargetNetwork(session):
    session.run(copy_Otarget)

def sample_final_epsilon():
    """
    Actor-learner helper
    Sample a final epsilon value to anneal towards
    from a discrete distribution.
    Called by each actor-learner thread.
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def build_initial_state(observation, state_buffer):
    """
    Fill the state buffer with the initial observation
    """
    processed_frame = get_preprocessed_frame(observation)
    state_buffer = deque() # clear the state buffer
    for _ in xrange(AGENT_HISTORY_LENGTH-1):
        state_buffer.append(processed_frame)
    return state_buffer

def get_state(observation, state_buffer):
    # Preprocess the current frame
    processed_frame = get_preprocessed_frame(observation)

    # Build state by concatenating current processed frame with previous
    # (agent_history_length-1) in the state buffer.
    state = np.empty((AGENT_HISTORY_LENGTH, RESIZED_HEIGHT, RESIZED_WIDTH))
    previous_frames = np.array(state_buffer)
    state[:AGENT_HISTORY_LENGTH-1, ...] = previous_frames
    state[AGENT_HISTORY_LENGTH-1] = processed_frame
    
    # Pop the oldest frame, add the current frame to the queue
    state_buffer.popleft()
    state_buffer.append(processed_frame), 
    state = np.reshape(state, (84,84,4))
    return state, state_buffer

def get_preprocessed_frame(observation):
    """
    See Methods->Preprocessing in Mnih et al.
    1) Get image grayscale
    2) Rescale image
    Note: Cannot take the max over the current and previous frame
    as they do in the paper, because gym skips a random amount of
    frames on each .act. Maybe try a workaround if performance
    really suffers.
    """
    return resize(rgb2gray(observation), (RESIZED_HEIGHT, RESIZED_WIDTH))

def actorLearner(num, env, session, lock):
    # We use global shared O parameter vector
    # We use global shared Otarget parameter vector
    # We use global shared counter T, and TMAX constant
    global TMAX, T

    # Initialize network gradients
    s_j_batch = []
    a_batch = []
    y_batch = []

    FINAL_EPSILON = sample_final_epsilon()
    epsilon = INITIAL_EPSILON

    state_buffer = deque()

    t = 0

    for i_episode in xrange(NUM_EPISODES):
        # Get initial game observation
        observation = env.reset()
        # Have actor-learner build up it's initial state
        state_buffer = build_initial_state(observation, state_buffer)

        reward = 0
        terminal = False
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0
        start_time = time.time()

        print "In THREAD ", num

        while True:
            # Actor-learner chooses action based on e-greedy policy
            """
            Choose action based on current state and e-greedy policy
            Returns action index and state.
            """
            # Get current state
            s_t, state_buffer = get_state(observation, state_buffer)
            # Forward the deep q network, get Q(s,a) values
            readout_t = O_readout.eval(session = session, feed_dict = {s : [s_t]})
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
    
            # Scale down epsilon
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    
            # Gym excecutes action in game environment on behalf of actor-learner
            # lock.acquire()
            observation, r_t, terminal, info = env.step(action_index)
            # lock.release()
            
            s_t1, state_buffer = get_state(observation, state_buffer)

            # Accumulate gradients
            readout_j1 = Ot_readout.eval(session = session, feed_dict = {st : [s_t1]})
            if terminal:
                y_batch.append(r_t)
            else:
                y_batch.append(r_t + GAMMA * np.max(readout_j1))
    
            a_batch.append(a_t)
            s_j_batch.append(s_t)
    
            # Update the old values
            T += 1
            t += 1
            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # Update the Otarget network
            if T % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                copyTargetNetwork(session)
    
            # Update the O network
            if t % NETWORK_UPDATE_FREQUENCY == 0 or terminal:
                if s_j_batch:
                    # Perform asynchronous update of O network
                    train_O.run(session = session, feed_dict = {
            	           y : y_batch,
            	           a : a_batch,
            	           s : s_j_batch})
    
                #Clear gradients
                s_j_batch = []
                a_batch = []
                y_batch = []
    
            # Save progress every 5000 iterations
            if t % CHECKPOINT_INTERVAL == 0:
                saver.save(session, CHECKPOINT_SAVE_PATH, global_step = t)
    
            # Print info
            state = ""
            if t < EXPLORE:
                state = "explore"
            else:
                state = "train"

            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(summary_vars)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print "THREAD:", num, "/ TIME", T, "/ TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ SCORE", ep_reward
                break


# We create the shared global networks
# O network
s, O_readout, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2 = createNetwork()
network_params = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

# Training node
a = tf.placeholder("float", [None, ACTIONS])
y = tf.placeholder("float", [None])
O_readout_action = tf.reduce_sum(tf.mul(O_readout, a), reduction_indices=1)
cost_O = tf.reduce_mean(tf.square(y - O_readout_action))
train_O = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(cost_O)

# Otarget network
st, Ot_readout, W_conv1t, b_conv1t, W_conv2t, b_conv2t, W_fc1t, b_fc1t, W_fc2t, b_fc2t = createNetwork()
copy_Otarget = [W_conv1t.assign(W_conv1), b_conv1t.assign(b_conv1), W_conv2t.assign(W_conv2), b_conv2t.assign(b_conv2), W_fc1t.assign(W_fc1), b_fc1t.assign(b_fc1), W_fc2t.assign(W_fc2), b_fc2t.assign(b_fc2)]

# Set up some episode summary ops to visualize on tensorboard.
episode_reward = tf.Variable(0.)
tf.scalar_summary("Episode Reward", episode_reward)
episode_ave_max_q = tf.Variable(0.)
tf.scalar_summary("Max Q Value", episode_ave_max_q)
logged_epsilon = tf.Variable(0.)
tf.scalar_summary("Epsilon", logged_epsilon)
logged_T = tf.Variable(0.)
tf.scalar_summary("T", logged_T)
logged_num_episodes = tf.Variable(0.)
tf.scalar_summary("Num Episodes", logged_num_episodes)
logged_time_per_episode = tf.Variable(0.)
tf.scalar_summary("Time per episode", logged_time_per_episode)

summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]

summary_op = tf.merge_all_summaries()

# Initialize session and variables
session = tf.InteractiveSession()
saver = tf.train.Saver(network_params)
session.run(tf.initialize_all_variables())
writer = tf.train.SummaryWriter(SUMMARY_SAVE_PATH, session.graph)

# checkpoint = tf.train.get_checkpoint_state("save_networks_asyn")
# if checkpoint and checkpoint.model_checkpoint_path:
#     saver.restore(session, checkpoint.model_checkpoint_path)
#     print "Successfully loaded:", checkpoint.model_checkpoint_path

if __name__ == "__main__":
    # Start n concurrent actor threads
    lock = threading.Lock()
    threads = list()
    
    # Initialize target network weights
    copyTargetNetwork(session)
    
    envs = [gym.make(GAME) for i in range(NUM_CONCURRENT)]
    actor_learner_threads = [threading.Thread(target=actorLearner, args=(thread_id, envs[thread_id], session, lock,)) for thread_id in range(NUM_CONCURRENT)]
    for t in actor_learner_threads:
        t.start()

    while True:
        # envs[0].render()
        for env in envs:
            env.render()
        summary_str = session.run(summary_op)
        writer.add_summary(summary_str, float(T))

    for t in actor_learner_threads:
        t.join()

    print "ALL DONE!!"
