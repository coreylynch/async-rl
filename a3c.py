#!/usr/bin/env python
from skimage.transform import resize
from skimage.color import rgb2gray
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
from a3c_model import build_policy_and_value_networks
from keras import backend as K
from atari_environment import AtariEnvironment

# Path params
EXPERIMENT_NAME = "pong_a3c"
SUMMARY_SAVE_PATH = "/Users/coreylynch/dev/async-rl/summaries/"+EXPERIMENT_NAME
# SUMMARY_SAVE_PATH = "/home/ec2-user/async-rl/summaries/"+EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = "/tmp/"+EXPERIMENT_NAME+".ckpt"
# CHECKPOINT_NAME = "/Users/coreylynch/dev/async-rl/ec2_checkpoints/pong_large.ckpt-310000"
CHECKPOINT_INTERVAL=5000
SUMMARY_INTERVAL=5
# TRAINING = False
TRAINING = True

SHOW_TRAINING = True
# SHOW_TRAINING = False

# Experiment params
GAME = "Pong-v0"
ACTIONS = 6
NUM_CONCURRENT = 8
NUM_EPISODES = 20000

AGENT_HISTORY_LENGTH = 4
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84

# DQN Params
GAMMA = 0.99

# Optimization Params
LEARNING_RATE = 7e-4
RMSPROP_DECAY = 0.99
RMSPROP_EPSILON = 0.1

#Shared global parameters
T = 0
TMAX = 80000000
t_max = 5

def sample_policy(num_actions, probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    probs = probs - np.finfo(np.float32).epsneg

    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index

def actor_learner_thread(num, env, session, graph_ops, summary_ops, saver):
    # We use global shared counter T, and TMAX constant
    global TMAX, T

    # Unpack graph ops
    s = graph_ops["s"]
    a = graph_ops["a"]
    R = graph_ops["R"]
    p_network = graph_ops["p_network"]
    p_grad_update = graph_ops["p_grad_update"]
    V = graph_ops["V"]
    v_grad_update = graph_ops["v_grad_update"]

    summary_placeholder, update_ep_reward, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT, agent_history_length=AGENT_HISTORY_LENGTH)
    s_t = env.get_initial_state()

    time.sleep(3*num)

    # Set up per-episode counters
    ep_reward = 0
    ep_avg_max_v = 0
    ep_t = 0

    while T < TMAX:
        # Clear gradients
        s_batch = []
        a_batch = []
        rewards = []
        t = 0
        while True:
            # Perform action according to policy
            probs = session.run(p_network, feed_dict={s: [s_t]})[0]
            if T % 10 == 0:
                print "max p, ", np.max(probs)
            action_index = sample_policy(ACTIONS, probs)
            a_t = np.zeros([ACTIONS])
            a_t[action_index] = 1

            s_batch.append(s_t)
            a_batch.append(a_t)
            
            s_t1, r_t, terminal, info = env.step(action_index)
            ep_reward += r_t
            
            r_t = np.clip(r_t, -1, 1)
            rewards.append(r_t)

            s_t = s_t1
            T += 1
            t += 1
            ep_t += 1

            if (terminal or (t == t_max)):
                break

        if terminal:
            R_t = 0
        else:
            R_t = session.run(V, feed_dict={s: [s_t]})[0][0] # Bootstrap from last state

        R_batch = np.zeros(t)
        for i in reversed(range(t)):
            R_t = rewards[i] + GAMMA * R_t
            R_batch[i] = R_t
            if T % 1000 == 0:
                print "V_t, ", R_t

        # Update policy network
        p_grad_update.run(session = session, feed_dict = {R : R_batch,
                                                          a : a_batch,
                                                          s : s_batch})
        # Update value network
        v_grad_update.run(session = session, feed_dict = {R : R_batch,
                                                          s : s_batch})

        if terminal:
            # Episode ended, collect stats and reset game
            session.run(update_ep_reward, feed_dict={summary_placeholder: ep_reward})
            print "THREAD:", num, "/ TIME", T, "/ REWARD", ep_reward
            s_t = env.get_initial_state()
            # Set up per-episode counters
            ep_reward = 0
            ep_avg_max_v = 0
            ep_t = 0

def build_graph():
    # Rewards
    R = tf.placeholder("float", [None])

    # One-hot vector representing chosen action
    a = tf.placeholder("float", [None, ACTIONS])

    # Create policy and value networks
    s, p_network, V = build_policy_and_value_networks(num_actions=ACTIONS, agent_history_length=AGENT_HISTORY_LENGTH, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT)

    # Policy loss
    log_prob = tf.log(tf.reduce_sum(tf.mul(p_network, a), reduction_indices=1))
    p_loss = -log_prob * (R - V)

    shared_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "shared")
    policy_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           "policy")
    value_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           "value")

    p_grad_update = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(p_loss, var_list=shared_network_params+policy_network_params)

    # Value loss
    v_loss = tf.reduce_mean(tf.square(R - V))
    v_grad_update = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(v_loss, var_list=shared_network_params+value_network_params)

    graph_ops = {"s" : s,
                 "a" : a,
                 "R" : R,
                 "p_network" : p_network,
                 "p_grad_update" : p_grad_update,
                 "V" : V,
                 "v_grad_update" : v_grad_update}

    return graph_ops

# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    summary_placeholder = tf.placeholder("float")
    update_ep_reward = episode_reward.assign(summary_placeholder)
    summary_op = tf.merge_all_summaries()
    return summary_placeholder, update_ep_reward, summary_op

def train(session, graph_ops, saver):
    # Set up game environments (one per thread)
    envs = [gym.make(GAME) for i in range(NUM_CONCURRENT)]
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[2]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_SAVE_PATH, session.graph)

    # Start NUM_CONCURRENT training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, envs[thread_id], session, graph_ops, summary_ops, saver)) for thread_id in range(NUM_CONCURRENT)]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        if SHOW_TRAINING:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > SUMMARY_INTERVAL:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()

def evaluation(session, graph_ops, saver):
    saver.restore(session, CHECKPOINT_NAME)
    print "Restored model weights from ", CHECKPOINT_NAME
    monitor_env = gym.make(GAME)
    monitor_env.monitor.start('/tmp/'+EXPERIMENT_NAME+"/eval")

    # Unpack graph ops
    s, q_values, network_params, st, target_q_values, target_network_params, reset_target_network_params, a, y, grad_update = graph_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT, agent_history_length=AGENT_HISTORY_LENGTH)

    for i_episode in xrange(100):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print ep_reward
    monitor_env.monitor.close()

def main(_):
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    K.set_session(session)
    graph_ops = build_graph()
    saver = tf.train.Saver()

    if TRAINING:
        train(session, graph_ops, saver)
    else:
        evaluation(session, graph_ops, saver)

if __name__ == "__main__":
  tf.app.run()
