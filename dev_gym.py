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
NUM_CONCURRENT = 5
NUM_EPISODES = 20
MAX_STEPS = 100
T_MAX = 250000 * 200
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

  def _build_graph(self):
    """ Build the graph describing shared networks, loss, optimizer, etc.
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
    # self._increment_T = self.T.assign_add(1)
    # self._reset_T = self.T.assign(0)
    # self._epoch = tf.Variable(0)
    # self._increment_epoch = self._epoch.assign_add(1)
    
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

    # self._state = state
    # self._action = action
    # self._reward = reward
    # self._new_state = new_state
    # self._terminal = terminal

    # self._network = q_values
    # self._async_gradient_update = grad_update

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
    Actor-learner helper
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
    
    for i_episode in xrange(NUM_EPISODES):
      # Get initial game observation
      observation = env.reset()
      # Have actor-learner build up it's initial state
      actor_learner.build_initial_state(observation)
      reward = 0
      done = False
      ep_reward = 0
      for j in xrange(MAX_STEPS):
        # Actor-learner selects action based on e-greedy policy
        action = actor_learner.act(observation, reward, done)
        print action
        
        # Gym excecutes action in game environment on behalf of actor-learner
        observation, reward, done, info = env.step(action)
        
        ep_reward += reward
        print ep_reward
        if done:
            print ep_reward
            print "Episode finished after {} timesteps".format(j+1)
            break

    # # Get initial game state
    # state = environment.get_initial_state()

    # epsilon = INITIAL_EXPLORATION
    # FINAL_EXPLORATION = self._sample_final_epsilon()

    # print "Thread ", i, "FINAL_EXPLORATION: ", FINAL_EXPLORATION

    # # Buffer to keep NETWORK_UPDATE_FREQUENCY recent state transitions
    # recent_history = deque()

    # # Counters that are reset after every episode (could potentially move this to Environment)
    # episode_reward = 0
    # episode_max_q_value = 0
    # episode_steps = 0

    # # Main learning loop
    # t = 0
    # start_time = time.time()
    # while True:
    #   # Get Q(s,a) values
    #   Q_s_a = self._session.run(self._network, feed_dict={self._state : state})[0]

    #   # Take action a according to the e-greedy policy
    #   if rng.rand() < epsilon:
    #     a = rng.randint(0, environment.num_actions())
    #   else:
    #     a =  np.argmax(Q_s_a)

    #   # Get current global timestep
    #   T = self._session.run(self.T)

    #   # Scale down epsilon
    #   if epsilon > FINAL_EXPLORATION:
    #     epsilon = INITIAL_EXPLORATION - ((INITIAL_EXPLORATION-FINAL_EXPLORATION) * float(T)/FINAL_EXPLORATION_FRAME)

    #   # Execute the chosen action in the environment, observe new state, reward, and indicator for
    #   # whether the episode terminated.
    #   (reward, new_state, terminal) = environment.execute(a)
    #   episode_reward += reward
    #   reward = np.clip(reward, -1, 1)

    #   # Build a one hot action vector
    #   actions_one_hot = np.zeros(environment.num_actions())
    #   actions_one_hot[a]=1

    #   # If we have NETWORK_UPDATE_FREQUENCY (s, a, r, s', t) tuples, do an async update, otherwise
    #   # keep collecting tuples
    #   if len(recent_history) == NETWORK_UPDATE_FREQUENCY:
    #     states, actions, rewards, new_states, terminals = self._unpack_recent_history(recent_history)
    #     feed_dict = {self._state : states, 
    #                  self._action: actions, 
    #                  self._reward : rewards, 
    #                  self._new_state : new_states, 
    #                  self._terminal : terminals}
    #     self._session.run(self._async_gradient_update, feed_dict=feed_dict)

    #     # Clear the recent_history queue
    #     for _ in range(NETWORK_UPDATE_FREQUENCY):
    #       recent_history.pop()
    #   else:
    #     recent_history.append((state, actions_one_hot, reward, new_state, terminal))

    #   # Collect some stats
    #   episode_max_q_value += np.max(Q_s_a)
    #   episode_steps += 1

    #   # s = s'
    #   if not terminal:
    #     state = new_state
    #   else:
    #     # Episode ended, so next state is the next episode's initial state
    #     state = environment.get_initial_state() # starts new episode

    #     # Log some end-of-episode stats
    #     self._log_end_of_episode_stats([episode_reward, episode_max_q_value/episode_steps, epsilon, T, time.time() - start_time])        
    #     start_time = time.time()
    #     episode_reward = 0
    #     episode_max_q_value = 0 
    #     episode_steps = 0

    #   self._session.run(self._increment_T) # T += 1
    #   t += 1

    #   # Optionally update the target network
    #   if T % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
    #     self._session.run(self._reset_target_network_params)

    #   if T >= T_MAX:
    #     break

  def train(self):
    """
    Creates and kicks off num_concurrent actor-learner threads
    """
    workers = []
    for thread_id in xrange(NUM_CONCURRENT):
      t = threading.Thread(target=self._actor_learner_thread, args=(thread_id,))
      t.start()
      workers.append(t)

    # time.sleep(60) # let the threads start
    # start_time = time.time()

    # while True:
    #   T = self._session.run(self.T)
    #   summary_str = self._session.run(self.summary_op)
    #   self.writer.add_summary(summary_str, float(T))
    #   if T >= T_MAX:
    #     break

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