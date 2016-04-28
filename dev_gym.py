import tensorflow as tf
import gym
import threading
import numpy as np
from collections import deque
from model import build_network, loss
import time
from keras import backend as K
from actor_learner import ActorLearner

# Path params
EXPERIMENT_NAME = "dev_gym"
SUMMARY_SAVE_PATH = "/Users/coreylynch/dev/async-rl/summaries/"+EXPERIMENT_NAME

# Experiment params
GAME = "Pong-v0"
ACTIONS = 6
NUM_CONCURRENT = 8
T_MAX = 250000 * 200
MAX_STEPS = 500

NUM_EPISODES = 20000
# NUM_EPISODES = T_MAX / NUM_CONCURRENT / MAX_STEPS

AGENT_HISTORY_LENGTH = 4
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84

# Async params
NETWORK_UPDATE_FREQUENCY = 5
TARGET_NETWORK_UPDATE_FREQUENCY = 40000

# DQN Params
GAMMA = 0.99
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

# Epsilon params
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION_FRAME = 4000000

class GlobalThread(object):
    """
    Main thread responsible for:
      - Initializing shared global network and target network parameters
      - Kicking off actor-learner threads
    """
    def __init__(self, session, graph):
        self._session = session
        self._graph = graph
        self._build_graph()
        self._session.run(self.graph_ops['reset_target_network'])
        self._envs = [gym.make(GAME) for i in range(NUM_CONCURRENT)]
        for env in self._envs:
            env.monitor.start('/tmp/'+EXPERIMENT_NAME)
        self._finished = [False for i in range(NUM_CONCURRENT)]

    def _build_graph(self):
        """ 
        Build the graph describing shared networks, loss, optimizer, etc.
        """
        # Input Placeholders
        state = tf.placeholder("float", [None, AGENT_HISTORY_LENGTH, RESIZED_WIDTH, RESIZED_HEIGHT])
        new_state = tf.placeholder("float", [None, AGENT_HISTORY_LENGTH, RESIZED_WIDTH, RESIZED_HEIGHT])
        action = tf.placeholder("float", [None, ACTIONS]) # One hot vector representing chosen action at time t
        reward = tf.placeholder("float", [None]) # instantaneous reward
        terminal = tf.placeholder("float", [None]) # 1 if terminal state, 0 otherwise

        # Network and target network
        q_values = build_network(state, "network_params", ACTIONS) # Q(s,a)
        target_q_values = build_network(new_state, "target_network_params", ACTIONS) # Q-(s',a)

        # Set up cost as a function of (s, a, r, s', terminal) tuples.
        cost = loss(q_values, target_q_values, action, reward, terminal)

        # Op for periodically updating target network with online network weights.
        network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         "network_params")
        target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "target_network_params")    
        # self._reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
        reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

        # Op for asyncronously computing/applying gradients.
        optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT, use_locking=False)
        grad_update = optimizer.minimize(cost, var_list=network_params)
        
        # Global counter variables
        T = tf.Variable(0)
        increment_T = T.assign_add(1)
        
        # Share these placeholders/methods with the actor-learner threads
        self.graph_ops = {'network': q_values,
                          'target_network': target_q_values,
                          'state': state,
                          'action': action, 
                          'reward': reward, 
                          'new_state': new_state,
                          'terminal': terminal,
                          'async_gradient_update': grad_update,
                          'reset_target_network': reset_target_network_params,
                          'T': T,
                          'increment_T': increment_T
                          }

        self._setup_counters()

        tf.initialize_all_variables().run()

    def _setup_counters(self):
      self._episode_reward_placeholder = tf.placeholder(tf.int32)
      episode_reward = tf.Variable(0)
      tf.scalar_summary("Episode Reward", episode_reward)
      self._log_episode_reward = episode_reward.assign(self._episode_reward_placeholder)

      self._episode_ave_max_q_placeholder = tf.placeholder("float")
      episode_ave_max_q = tf.Variable(0.)
      tf.scalar_summary("Max Q Value", episode_ave_max_q)
      self._log_episode_ave_max_q = episode_ave_max_q.assign(self._episode_ave_max_q_placeholder)

      self._log_epsilon_placeholder = tf.placeholder("float")
      logged_epsilon = tf.Variable(0.)
      tf.scalar_summary("Epsilon", logged_epsilon)
      self._log_epsilon = logged_epsilon.assign(self._log_epsilon_placeholder)

      self._log_T_placeholder = tf.placeholder(tf.int32)
      logged_T = tf.Variable(0)
      tf.scalar_summary("T", logged_T)
      self._log_T = logged_T.assign(self._log_T_placeholder)

      self._log_time_per_episode_placeholder = tf.placeholder("float")
      logged_time_per_episode = tf.Variable(0.)
      tf.scalar_summary("Time per episode", logged_time_per_episode)
      self._log_time_per_episode = logged_time_per_episode.assign(self._log_time_per_episode_placeholder)

      summary_op = tf.merge_all_summaries()
      self.summary_op = summary_op
      writer = tf.train.SummaryWriter(SUMMARY_SAVE_PATH, self._session.graph)
      self.writer = writer

    def _log_end_of_episode_stats(self, stats):
      self._session.run(self._log_episode_reward, feed_dict={self._episode_reward_placeholder : stats[0]})
      self._session.run(self._log_episode_ave_max_q, feed_dict={self._episode_ave_max_q_placeholder : stats[1]})
      self._session.run(self._log_epsilon, feed_dict={self._log_epsilon_placeholder : stats[2]})
      self._session.run(self._log_T, feed_dict={self._log_T_placeholder : stats[3]})
      self._session.run(self._log_time_per_episode, feed_dict={self._log_time_per_episode_placeholder : stats[4]})

    def _sample_final_epsilon(self):
      """
      Actor-learner helper
      Sample a final epsilon value to anneal towards
      from a discrete distribution.
      Called by each actor-learner thread.
      """
      final_epsilons = np.array([.1,.01,.5])
      probabilities = np.array([0.4,0.3,0.3])
      return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

    def _unpack_recent_history(self, recent_history):
      """
      Unpack the recent_history tuples into separate numpy array variables 
      """
      states = np.reshape(np.array([h[0] for h in recent_history]), (NETWORK_UPDATE_FREQUENCY, AGENT_HISTORY_LENGTH, RESIZED_WIDTH, RESIZED_HEIGHT))
      actions = np.array([h[1] for h in recent_history])
      rewards = np.array([h[2] for h in recent_history])
      new_states = np.reshape(np.array([h[3] for h in recent_history]), (NETWORK_UPDATE_FREQUENCY, AGENT_HISTORY_LENGTH, RESIZED_WIDTH, RESIZED_HEIGHT))
      terminals = np.array([h[4] for h in recent_history])
      return (states, actions, rewards, new_states, terminals)

    def _actor_learner_thread(self, i):
        """
        Actor learner thread:
        - Initialize thread-specific environment
        - Run a training loop, periodically
          sending asyncronous gradient updates to main model
        """
        # Initialize actor-learner
        env = self._envs[i]
  
        final_epsilon = self._sample_final_epsilon()
        actor_learner = ActorLearner(env.action_space, 
                                     self._session, 
                                     self.graph_ops, 
                                     final_epsilon,
                                     FINAL_EXPLORATION_FRAME)
        
        # Buffer to keep NETWORK_UPDATE_FREQUENCY recent state transitions
        recent_history = deque()
        
        for i_episode in xrange(NUM_EPISODES):
            # Get initial game observation
            observation = env.reset()
            # Have actor-learner build up it's initial state
            actor_learner.build_initial_state(observation)
            reward = 0
            done = False
            ep_reward = 0
            episode_ave_max_q = 0
    
            t = 0
            start_time = time.time()
            while True:
                # Actor-learner chooses action based on e-greedy policy
                (action, state, max_q_value) = actor_learner.choose_action(observation, reward, done)
                
                # Gym excecutes action in game environment on behalf of actor-learner
                observation, reward, done, info = env.step(action)
                
                # Collect some stats
                ep_reward += reward
                episode_ave_max_q += max_q_value
        
                # T+=1
                T = self._session.run(self.graph_ops["T"])
                self._session.run(self.graph_ops["increment_T"])
                t += 1
        
                # If we have NETWORK_UPDATE_FREQUENCY (s, a, r, s', t) tuples, do an async update, otherwise
                # keep collecting tuples
                if len(recent_history) == NETWORK_UPDATE_FREQUENCY:
                      states, actions, rewards, new_states, terminals = self._unpack_recent_history(recent_history)
                      feed_dict = {self.graph_ops["state"] : states, 
                                   self.graph_ops["action"] : actions, 
                                   self.graph_ops["reward"] : rewards, 
                                   self.graph_ops["new_state"] : new_states, 
                                   self.graph_ops["terminal"] : terminals}
                      self._session.run(self.graph_ops["async_gradient_update"], feed_dict=feed_dict)
              
                      # Clear the recent_history queue
                      [recent_history.pop() for _ in range(NETWORK_UPDATE_FREQUENCY)]
                else:
                      # Build a new transition tuple (s, a, r, s', t)
                      new_state = actor_learner.get_state(observation)
                      
                      # Clip reward
                      reward = np.clip(reward, -1, 1)
    
                      done = 1. if done else 0.
                      actions_one_hot = np.zeros(env.action_space.n)
                      actions_one_hot[action]=1
      
                      transition = (state, action, reward, new_state, done)
      
                      recent_history.append((state, actions_one_hot, reward, new_state, done))      
        
                # Optionally update the target network
                if T % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    self._session.run(self.graph_ops["reset_target_network"])
    
                if (t > MAX_STEPS) or done:
                    # Episode ended, report some stats
                    self._log_end_of_episode_stats([ep_reward, episode_ave_max_q/t, actor_learner.epsilon, T, time.time() - start_time])
                    break

        self._finished[i]=True # Global thread finishes and cleans up when all actor-learner threads signal they're done


    def train(self):
        """
        Creates and kicks off num_concurrent actor-learner threads
        """
        workers = []
        for thread_id in xrange(NUM_CONCURRENT):
          t = threading.Thread(target=self._actor_learner_thread, args=(thread_id,))
          t.start()
          workers.append(t)

        time.sleep(10) # let the threads start

        while True:
            self._envs[0].render()
            T = self._session.run(self.graph_ops["T"])
            summary_str = self._session.run(self.summary_op)
            self.writer.add_summary(summary_str, float(T))
            if np.all(self._finished):
                for env in self._envs:
                    env.monitor.close()
                break

        for t in workers:
          t.join()
        # self._actor_learner_thread(0) # debug

def main(_):
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      K.set_session(session) # set Keras session
      global_thread = GlobalThread(session, g)
      global_thread.train()

if __name__ == "__main__":
  tf.app.run()