import tensorflow as tf
from ale_python_interface import ALEInterface
import threading
from skimage.transform import resize
import numpy as np
from collections import deque
from model import build_network2, loss
from environment import Environment

STEPS_PER_EPOCH = 250000
EPOCHS = 200
STEPS_PER_TEST = 125000

ACTIONS = 4
NUM_CONCURRENT = 16
GAMMA = 0.99
NETWORK_UPDATE_FREQUENCY = 5
TARGET_NETWORK_UPDATE_FREQUENCY = 10000

LEARNING_RATE = 0.00025

GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

NOOP_MAX = 30
RESIZED_HEIGHT = 84
RESIZED_WIDTH = 84
AGENT_HISTORY_LENGTH = 4

INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION_FRAME = 4000000 # Section 5.1 says "the values of [epsilon] were annealed ... over the first four million frames"

class GlobalThread(object):
  """
  Main thread responsible for:
    - Initializing shared global network and target network parameters
    - Kicking off actor-learner threads
  """
  def __init__(self, session, graph, ale_io_lock):
    self._session = session
    self._graph = graph
    self._ale_io_lock = ale_io_lock
    self._build_graph()
    self.rng = np.random.RandomState(123456)
    self._session.run(self._reset_target_network_params)
  
  def _build_graph(self):
    """ Placeholder for the function that builds up the model 
    """
    # Input Placeholders
    state = tf.placeholder("float", [None, RESIZED_WIDTH, RESIZED_HEIGHT, AGENT_HISTORY_LENGTH]) # Previous agent_history_length frames
    a = tf.placeholder("float", [None, ACTIONS]) # One hot vector representing chosen action at time t
    y = tf.placeholder("float", [None]) # Float holding y_t, the target value for chosen action at time t

    # Set up network and target network
    with tf.variable_scope('network_params') as scope:
      network = build_network2(state)
    with tf.variable_scope('target_network_params') as scope:
      target_network = build_network2(state)
    
    # Set up loss
    cost = loss(network, a, y)

    # Op for periodically updating target network with online network weights
    network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "network_params")
    target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "target_network_params")
    self._reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

    # Global ops for doing async apply gradients
    # Basically, set up a shared global op that applies placeholder gradients,
    # then have each actor-learner call that op asyncronously, feeding in their accumulated gradients
    # as the placeholder gradients.
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum=GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT, use_locking=False)
    compute_gradients = optimizer.compute_gradients(cost, var_list=network_params)
    
    placeholder_gradients = []
    for grad_var in compute_gradients:
       placeholder_gradients.append((tf.placeholder('float', shape=grad_var[1].get_shape()), grad_var[1]))

    self._async_apply_gradients = optimizer.apply_gradients(placeholder_gradients)
    
    # Global counter variable
    T = tf.Variable(0)
    self.T = T
    self._increment_T = T.assign_add(1)

    # Share these with the actor-learner threads
    self._state = state
    self._a = a
    self._y = y
    self._network = network
    self._network_params = network_params
    self._target_network = target_network
    self._cost = cost
    self._compute_gradients = compute_gradients
    self._placeholder_gradients = placeholder_gradients
    

    tf.initialize_all_variables().run()
    self.saver = tf.train.Saver()

  def _sample_final_epsilon(self):
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

  def _actor_learner_thread(self, i):
    """
    Actor learner thread:
    - Initialize thread-specific environment
    - Run a training loop, periodically
      sending asyncronous gradient updates to main model
    """
    # Initialize this thread's agent's environment
    environment = Environment(self._ale_io_lock)
    
    # Get initial game state
    state = environment.get_initial_state()

    epsilon = INITIAL_EXPLORATION
    FINAL_EXPLORATION = self._sample_final_epsilon()

    print "Thread ", i, "FINAL_EXPLORATION: ", FINAL_EXPLORATION

    # Thread specific gradients
    accum_gradients = np.array([np.zeros(var.get_shape().as_list(), dtype=np.float32) for var in self._network_params])

    # Main learning loop
    T_max = 50000000
    t = 0
    episode_reward = 0
    episode_max_q_value = 0
    episode_steps = 0
    episode_cost = 0
    while self._session.run(self.T) < T_max:
      # Get Q(s,a) values
      Q_s_a = self._session.run(self._network, feed_dict={self._state : state })

      # Take action a according to the e-greedy policy
      if self.rng.rand() < epsilon:
        action = self.rng.randint(0, environment.num_actions())
      else:
        action =  np.argmax(Q_s_a)

      # Execute the chosen action in the environment, observe new state and reward.
      (reward, new_state, terminal) = environment.execute(action)
      episode_reward += reward
      reward = np.clip(reward, -1, 1)

      # Compute y using target network
      target_Q_s_a = self._session.run(self._target_network, feed_dict={self._state : new_state })
      if terminal:
        y = reward
      else:
        y = reward + GAMMA * np.max(target_Q_s_a)

      # Build a one hot action vector
      actions_one_hot = np.zeros(environment.num_actions())
      actions_one_hot[action]=1.0

      # Accumulate gradients
      grad_vals = self._session.run([grad for grad, _ in self._compute_gradients], feed_dict={self._state : state, self._a: [actions_one_hot], self._y: [y]})
      # print np.max(Q_s_a)
      # print y
      accum_gradients += grad_vals

      # Report cost
      cost = self._session.run(self._cost, feed_dict={self._state : state, self._a: [actions_one_hot], self._y: [y]})
      episode_cost += cost

      # Collect some stats (TODO: put in tensorflow summaries)
      episode_max_q_value += np.max(Q_s_a)
      episode_steps += 1

      # Scale down epsilon
      if epsilon > FINAL_EXPLORATION:
        epsilon -= (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME

      # s = s'
      if not terminal:
        state = new_state
      else:
        # Episode ended, so next state is the next episode's initial state
        state = environment.get_initial_state()

        # Print out some end-of-episode stats
        episode_average_max_q_value = episode_max_q_value / float(episode_steps)
        episode_average_cost = episode_cost / float(episode_steps)
        print "Reward: %i   Average Q(s,a): %.3f Cost: %.10f Epsilon: %.5f  Target epsilon: %.2f" % (episode_reward, episode_average_max_q_value, episode_average_cost, epsilon, FINAL_EXPLORATION)
        episode_reward = 0
        episode_max_q_value = 0 
        episode_cost = 0
        episode_steps = 0


      # T += 1
      self._session.run(self._increment_T)
      T = self._session.run(self.T)
      t += 1

      # Update the target network
      if T % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
        self._session.run(self._reset_target_network_params)

      # Perform asynchronous update
      if (t % NETWORK_UPDATE_FREQUENCY == 0) or terminal:
        self.async_apply_gradients(accum_gradients)
        accum_gradients *= 0 # Zero out accumulated gradients

  def async_apply_gradients(self, accum_gradients):
    """
    Gets gradients as numpy arrays,
    Runs shared async_apply_gradients handing in
    numpy gradients to the placeholders
    """
    feed_dict = {}
    for j, (grad, var) in enumerate(self._compute_gradients):
      feed_dict[self._placeholder_gradients[j][0]] = accum_gradients[j]
    self._session.run(self._async_apply_gradients, feed_dict=feed_dict)

  def train(self):
    """
    Creates and kicks off num_concurrent actor-learner threads
    """
    workers = []
    for thread_id in xrange(NUM_CONCURRENT):
      t = threading.Thread(target=self._actor_learner_thread, args=(thread_id,))
      t.start()
      workers.append(t)
    for t in workers:
      t.join()

def main(_):
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      ale_io_lock = threading.Lock() # using a lock to avoid race condition on ALE init
      global_thread = GlobalThread(session, g, ale_io_lock)
      for _ in xrange(EPOCHS):
        global_thread.train()

if __name__ == "__main__":
  tf.app.run()