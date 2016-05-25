#!/usr/bin/env python
from skimage.transform import resize
from skimage.color import rgb2gray
from atari_environment import AtariEnvironment
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
from model import build_network
from keras import backend as K

# Path params
EXPERIMENT_NAME = "breakout"
SUMMARY_SAVE_PATH = "/Users/coreylynch/dev/async-rl/summaries/"+EXPERIMENT_NAME
# SUMMARY_SAVE_PATH = "/home/ec2-user/async-rl/summaries/"+EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = "/tmp/"+EXPERIMENT_NAME+".ckpt"
# CHECKPOINT_NAME = "/Users/coreylynch/dev/async-rl/ec2_checkpoints/pong_large.ckpt-310000"
CHECKPOINT_INTERVAL=5000
SUMMARY_INTERVAL=5
# TRAINING = False
TRAINING = True

# SHOW_TRAINING = False
SHOW_TRAINING = True

# Experiment params
GAME = "Breakout-v0"
ACTIONS = 6
NUM_CONCURRENT = 8
NUM_EPISODES = 20000

AGENT_HISTORY_LENGTH = 4
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84

# Async params
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
NETWORK_UPDATE_FREQUENCY = 5

# DQN Params
GAMMA = 0.99

# Optimization Params
LEARNING_RATE = 7e-4
RMSPROP_DECAY = 0.99
RMSPROP_EPSILON = 0.1

# Epsilon params
INITIAL_EPSILON = 1.0
FINAL_EXPLORATION_FRAME = 1000000

#Shared global parameters
T = 0
TMAX = 80000000

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

def actor_learner_thread(num, env, session, graph_ops, summary_ops, saver):
    # We use global shared counter T, and TMAX constant
    global TMAX, T

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]
    learning_rate = graph_ops["learning_rate"]

    summary_placeholders, update_ops, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT, agent_history_length=AGENT_HISTORY_LENGTH)

    # Initialize network gradients
    s_j_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    epsilon = INITIAL_EPSILON

    print "Starting thread ", num, "with final epsilon ", final_epsilon 

    time.sleep(3*num)
    t = 0
    while T < TMAX:
        # Get initial game observation
        s_t = env.get_initial_state()
        terminal = False

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while True:
            # Actor-learner chooses action based on e-greedy policy
            """
            Choose action based on current state and e-greedy policy
            Returns action index and state.
            """
            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # Scale down epsilon
            if epsilon > final_epsilon:
                # epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - final_epsilon) * (T/float(FINAL_EXPLORATION_FRAME))
                epsilon -= (INITIAL_EPSILON - final_epsilon) / FINAL_EXPLORATION_FRAME
    
            # Gym excecutes action in game environment on behalf of actor-learner
            s_t1, r_t, terminal, info = env.step(action_index)

            # Accumulate gradients
            readout_j1 = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + GAMMA * np.max(readout_j1))
    
            a_batch.append(a_t)
            s_j_batch.append(s_t)
    
            # Update the old values
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # Update the Otarget network
            if T % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                session.run(reset_target_network_params)
    
            # Update the O network
            if t % NETWORK_UPDATE_FREQUENCY == 0 or terminal:
                if s_j_batch:
                    # Get current linearly annealed lr
                    lr = LEARNING_RATE * (TMAX-T)/float(TMAX)

                    # Perform asynchronous update of O network
                    grad_update.run(session = session, feed_dict = {
                           y : y_batch,
                           a : a_batch,
                           s : s_j_batch,
                           learning_rate : lr})
    
                # Clear gradients
                s_j_batch = []
                a_batch = []
                y_batch = []
    
            # Save progress every 5000 iterations
            if t % CHECKPOINT_INTERVAL == 0:
                saver.save(session, CHECKPOINT_SAVE_PATH, global_step = t)
    
            # Print info
            state = ""
            if t < FINAL_EXPLORATION_FRAME:
                state = "explore"
            else:
                state = "train"

            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print "THREAD:", num, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ PROGRESS", t/float(FINAL_EXPLORATION_FRAME)
                break

def build_graph():
    # Create shared deep q network
    s, q_network = build_network(num_actions=ACTIONS, agent_history_length=AGENT_HISTORY_LENGTH, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT)
    network_params = q_network.trainable_weights
    q_values = q_network(s)

    # Create shared target network
    st, target_q_network = build_network(num_actions=ACTIONS, agent_history_length=AGENT_HISTORY_LENGTH, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT)
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)

    # Op for periodically updating target network with online network weights.
    reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
    
    # Define cost and gradient update op
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - action_q_values))
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.RMSPropOptimizer(learning_rate, RMSPROP_DECAY, RMSPROP_DECAY, RMSPROP_EPSILON)
    grad_update = optimizer.minimize(cost, var_list=network_params, gate_gradients=optimizer.GATE_NONE)

    graph_ops = {"s" : s, 
                 "q_values" : q_values,
                 "st" : st, 
                 "target_q_values" : target_q_values,
                 "reset_target_network_params" : reset_target_network_params,
                 "a" : a,
                 "y" : y,
                 "grad_update" : grad_update,
                 "learning_rate" : learning_rate}

    return graph_ops

# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Max Q Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.scalar_summary("Epsilon", logged_epsilon)
    logged_T = tf.Variable(0.)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, update_ops, summary_op

def train(session, graph_ops, saver):
    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

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
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

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
