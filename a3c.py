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
EXPERIMENT_NAME = "breakout_a3c"
# SUMMARY_SAVE_PATH = "/Users/coreylynch/dev/async-rl/summaries/"+EXPERIMENT_NAME
SUMMARY_SAVE_PATH = "/home/ec2-user/async-rl/summaries/"+EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = "/tmp/"+EXPERIMENT_NAME+".ckpt"
# CHECKPOINT_NAME = "/Users/coreylynch/dev/async-rl/ec2_checkpoints/pong_large.ckpt-310000"
CHECKPOINT_INTERVAL=5000
SUMMARY_INTERVAL=5
# TRAINING = False
TRAINING = True

# SHOW_TRAINING = True
SHOW_TRAINING = False

# Experiment params
GAME = "Breakout-v0"
ACTIONS = 6
NUM_CONCURRENT = 16
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
    apply_grads, learning_rate, grads_and_vars, grads_placeholders, all_params, thread_ops = graph_ops
    ops = thread_ops[num]
    s = ops["s"]
    p_network = ops["p_network"]
    v_network = ops["v_network"]
    a = ops["a"]
    R = ops["R"]
    sync_p = ops["sync_p"]
    sync_v = ops["sync_v"]
    grads = ops["grads"]

    # Unpack tensorboard summary stuff
    r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT, agent_history_length=AGENT_HISTORY_LENGTH)
    s_t = env.get_initial_state()

    time.sleep(3*num)

    # Set up per-episode counters
    ep_reward = 0
    ep_avg_v = 0
    v_steps = 0
    ep_t = 0

    t = 0
    t_start = 0

    s_t = env.get_initial_state()
    terminal = False

    accum_gradients = np.array([np.zeros(var.get_shape().as_list(), dtype=np.float32) for var in all_params])

    while T < TMAX:
        # # Reset gradients
        accum_gradients *= 0.0
        past_states = {}
        past_rewards = {}
        past_actions = {}

        # # Sync policy and value networks
        session.run(sync_p)
        session.run(sync_v)

        t_start = t

        while not (terminal or ((t - t_start)  == t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            probs = session.run(p_network, feed_dict={s: [s_t]})[0]
            if T % 10 == 0:
                print "max p, ", np.max(probs)
            action_index = sample_policy(ACTIONS, probs)
            a_t = np.zeros([ACTIONS])
            a_t[action_index] = 1

            past_states[t] = s_t
            past_actions[t] = a_t

            s_t1, r_t1, terminal, info = env.step(action_index)
            ep_reward += r_t1

            t += 1
            T += 1
            ep_t += 1

            r_t1 = np.clip(r_t1, -1, 1)
            past_rewards[t-1] = r_t1
            
            s_t = s_t1

        if terminal:
            R_t = 0
        else:
            R_t = session.run(v_network, feed_dict={s: [s_t]})[0][0] # Bootstrap from last state

        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + GAMMA * R_t
            a_t = past_actions[i]
            s_t = past_states[i]
            g = session.run(grads, feed_dict={R : [R_t],
                                              a : [a_t],
                                              s : [s_t]})
            accum_gradients += g

        # Apply gradients remotely
        feed_dict = {}
        for i, grad_var in enumerate(grads_and_vars):
            feed_dict[grads_placeholders[i][0]] = accum_gradients[i]
        feed_dict[learning_rate] = (float(TMAX) - (T - 1))/ TMAX * LEARNING_RATE
        session.run(apply_grads, feed_dict=feed_dict)

        if terminal:
            # Episode ended, collect stats and reset game
            session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
            print "THREAD:", num, "/ TIME", T, "/ REWARD", ep_reward
            s_t = env.get_initial_state()
            terminal = False
            # Reset per-episode counters
            ep_reward = 0
            ep_t = 0

def copy_to(global_vars, thread_vars):
    """
    Copies global weights to thread weights
    """
    sync = [thread_vars[i].assign(global_vars[i]) for i in range(len(global_vars))]
    return sync

def union_params(p_params, v_params):
    union = set(p_params+v_params)
    union = [(i.name, i) for i in union]
    consistent_ordering = sorted(union, key=lambda tup: tup[0])
    return [i[1] for i in consistent_ordering]

def build_graph():

    # Create shared global policy and value networks
    s, p_network, v_network, p_params, v_params = build_policy_and_value_networks(num_actions=ACTIONS, agent_history_length=AGENT_HISTORY_LENGTH, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT)

    # Shared global optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.RMSPropOptimizer(learning_rate, RMSPROP_DECAY, RMSPROP_DECAY, RMSPROP_EPSILON)

    # One map of thread_name->thread_op per thread
    thread_ops = []
    
    for i in range(NUM_CONCURRENT):
        # One pair of (policy, value) networks per thread
        s_i, p_network_i, v_network_i, p_params_i, v_params_i = build_policy_and_value_networks(num_actions=ACTIONS, agent_history_length=AGENT_HISTORY_LENGTH, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT)

        # Op for synchronizing global params with thread params
        sync_p = copy_to(p_params, p_params_i)
        sync_v = copy_to(v_params, v_params_i)

        # Discounted future reward at time t
        R_t = tf.placeholder("float", [None])

        # One-hot vector representing chosen action at time t
        a_t = tf.placeholder("float", [None, ACTIONS])

        # Policy loss
        log_prob = tf.log(tf.reduce_sum(tf.mul(p_network_i, a_t), reduction_indices=1))
        p_loss = -log_prob * (R_t - v_network_i) # maximizing log prob

        # Value Loss
        v_loss = tf.reduce_mean(tf.square(R_t - v_network_i))

        total_loss = p_loss + (0.5 * v_loss)

        # One pair of (policy, value) compute_gradients per thread
        all_params_i = union_params(p_params_i, v_params_i)
        
        compute_gradients_i = optimizer.compute_gradients(total_loss, var_list=all_params_i, gate_gradients=optimizer.GATE_NONE)

        grads = [g for g, _ in compute_gradients_i]

        ops = {"s" : s_i,
               "p_network" : p_network_i,
               "v_network" : v_network_i,
               "R" : R_t,
               "a" : a_t,
               "grads" : grads,
               "sync_p" : sync_p,
               "sync_v" : sync_v}
        thread_ops.append(ops)
        
    # Op for applying remote gradients
    R_t = tf.placeholder("float", [None])
    a_t = tf.placeholder("float", [None, ACTIONS])
    log_prob = tf.log(tf.reduce_sum(tf.mul(p_network, a_t), reduction_indices=1))
    p_loss = -log_prob * (R_t - v_network)
    v_loss = tf.reduce_mean(tf.square(R_t - v_network))

    total_loss = p_loss + (0.5 * v_loss) # NOTE: this isn't actually used, just a prereq for setting up apply_gradients

    all_params = union_params(p_params, v_params)
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list=all_params, gate_gradients=optimizer.GATE_NONE)

    grads_placeholders = []
    for grad_var in grads_and_vars:
       grads_placeholders.append((tf.placeholder('float', shape=grad_var[1].get_shape()), grad_var[1]))

    apply_grads = optimizer.apply_gradients(grads_placeholders)

    return apply_grads, learning_rate, grads_and_vars, grads_placeholders, all_params, thread_ops

# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    r_summary_placeholder = tf.placeholder("float")
    update_ep_reward = episode_reward.assign(r_summary_placeholder)
    ep_avg_v = tf.Variable(0.)
    tf.scalar_summary("Episode Value", ep_avg_v)
    val_summary_placeholder = tf.placeholder("float")
    update_ep_val = ep_avg_v.assign(val_summary_placeholder)
    summary_op = tf.merge_all_summaries()
    return r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op

def train(session, graph_ops, saver):
    # Set up game environments (one per thread)
    envs = [gym.make(GAME) for i in range(NUM_CONCURRENT)]
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

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
