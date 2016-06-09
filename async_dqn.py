#!/usr/bin/env python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

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
from model import build_network
from keras import backend as K

flags = tf.app.flags

flags.DEFINE_string('experiment', 'dqn_breakout', 'Name of the current experiment')
flags.DEFINE_string('game', 'Breakout-v0', 'Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 80000000, 'Number of training timesteps.')
flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
flags.DEFINE_integer('agent_history_length', 4, 'Use this number of recent screens as the environment state.')
flags.DEFINE_integer('network_update_frequency', 32, 'Frequency with which each actor learner thread does an async gradient update')
flags.DEFINE_integer('target_network_update_frequency', 10000, 'Reset the target network every n timesteps')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, 'Number of timesteps to anneal epsilon.')
flags.DEFINE_string('summary_dir', '/tmp/summaries', 'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoints', 'Directory for storing model checkpoints')
flags.DEFINE_integer('summary_interval', 5,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval.')
flags.DEFINE_integer('checkpoint_interval', 600,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds (rounded up to statistics interval.')
flags.DEFINE_boolean('show_training', True, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', 'path/to/recent.ckpt', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_string('eval_dir', '/tmp/', 'Directory to store gym evaluation')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')
FLAGS = flags.FLAGS
T = 0
TMAX = FLAGS.tmax

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def actor_learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
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

    summary_placeholders, update_ops, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)

    # Initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print "Starting thread ", thread_id, "with final epsilon ", final_epsilon

    time.sleep(3*thread_id)
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
            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})
            
            # Choose next action based on e-greedy policy
            a_t = np.zeros([num_actions])
            action_index = 0
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps
    
            # Gym excecutes action in game environment on behalf of actor-learner
            s_t1, r_t, terminal, info = env.step(action_index)

            # Accumulate gradients
            readout_j1 = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + FLAGS.gamma * np.max(readout_j1))
    
            a_batch.append(a_t)
            s_batch.append(s_t)
    
            # Update the state and counters
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # Optionally update target network
            if T % FLAGS.target_network_update_frequency == 0:
                session.run(reset_target_network_params)
    
            # Optionally update online network
            if t % FLAGS.network_update_frequency == 0 or terminal:
                if s_batch:
                    session.run(grad_update, feed_dict = {y : y_batch,
                                                          a : a_batch,
                                                          s : s_batch})
                # Clear gradients
                s_batch = []
                a_batch = []
                y_batch = []
    
            # Save model progress
            if t % FLAGS.checkpoint_interval == 0:
                saver.save(session, FLAGS.checkpoint_dir+"/"+FLAGS.experiment+".ckpt", global_step = t)
    
            # Print end of episode stats
            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ EPSILON PROGRESS", t/float(FLAGS.anneal_epsilon_timesteps)
                break

def build_graph(num_actions):
    # Create shared deep q network
    s, q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height)
    network_params = q_network.trainable_weights
    q_values = q_network(s)

    # Create shared target network
    st, target_q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height)
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)

    # Op for periodically updating target network with online network weights
    reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
    
    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - action_q_values))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s" : s, 
                 "q_values" : q_values,
                 "st" : st, 
                 "target_q_values" : target_q_values,
                 "reset_target_network_params" : reset_target_network_params,
                 "a" : a,
                 "y" : y,
                 "grad_update" : grad_update}

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

def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    # Figure out number of actions from gym env
    env = gym.make(FLAGS.game)
    num_actions = env.action_space.n
    if (FLAGS.game == "Pong-v0" or FLAGS.game == "Breakout-v0"):
        # Gym currently specifies 6 actions for pong
        # and breakout when only 3 are needed. This
        # is a lame workaround.
        num_actions = 3
    return num_actions

def train(session, graph_ops, num_actions, saver):
    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    # Set up game environments (one per thread)
    envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
    writer = tf.train.SummaryWriter(summary_save_path, session.graph)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # Start num_concurrent actor-learner training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, envs[thread_id], session, graph_ops, num_actions, summary_ops, saver)) for thread_id in range(FLAGS.num_concurrent)]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        if FLAGS.show_training:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > FLAGS.summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()

def evaluation(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    print "Restored model weights from ", FLAGS.checkpoint_path
    monitor_env = gym.make(FLAGS.game)
    monitor_env.monitor.start(FLAGS.eval_dir+"/"+FLAGS.experiment+"/eval")

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)

    for i_episode in xrange(FLAGS.num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
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
    num_actions = get_num_actions()
    graph_ops = build_graph(num_actions)
    saver = tf.train.Saver()

    if FLAGS.testing:
        evaluation(session, graph_ops, saver)
    else:
        train(session, graph_ops, num_actions, saver)

if __name__ == "__main__":
  tf.app.run()
