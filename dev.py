
import tensorflow as tf
from ale_python_interface import ALEInterface
import threading
from skimage.transform import resize
import numpy as np
from collections import deque
from model import build_network, loss
from environment import Environment

ACTIONS = 4
NUM_CONCURRENT = 4 # 16
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
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000

class GlobalThread(object):
  """
  Main global thread responsible for:
    - Initializing shared global network and target network parameters
    - Kicking off actor-learner threads
  """
  def __init__(self, session, graph, ale_io_lock, lock):
    self._session = session
    self._graph = graph
    self._ale_io_lock = ale_io_lock
    self.build_graph()
    self.rng = np.random.RandomState(123456)
    self._session.run(self._reset_target_network_params)
    self.lock = lock
  
  def build_graph(self):
    """ Placeholder for the function that builds up the model 
    """
    # Input Placeholders
    state = tf.placeholder("float", [None, 84, 84, 4]) # Previous agent_history_length frames
    a = tf.placeholder("float", [None, ACTIONS]) # One hot vector representing chosen action at time t
    y = tf.placeholder("float", [None]) # Float holding y_t, the target value for chosen action at time t
    self._state=state
    self._a = a
    self._y = y

    # Set up online network and target network.
    with tf.variable_scope('network_params') as scope:
      network = build_network(state)
    with tf.variable_scope('target_network_params') as scope:
      target_network = build_network(state)
    self._network = network
    self._target_network = target_network
    
    # Set up loss
    cost = loss(network, a, y)
    self._cost = cost

    # Op for periodically updating target network with online network weights
    network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "network_params")
    target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "target_network_params")
    self._reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

    # One gradients list per thread
    for i in range(NUM_CONCURRENT):
      with tf.variable_scope('grads_'+str(i)) as scope:
        grad_vars = [tf.Variable(np.zeros(var.get_shape().as_list(), dtype=np.float32), trainable=False) for var in network_params]

    # One optimizer per thread    
    optimizers = []
    for i in range(NUM_CONCURRENT):
        optimizers.append(tf.train.AdamOptimizer(LEARNING_RATE))
    self._optimizers = optimizers

    # One gradient assign add (+=) op per thread
    # Assign_add computes gradients using the thread's optimizer,
    # then assigns the computed gradients to the thread's grad_vars
    assign_add_gradients_ops = []
    for i in range(NUM_CONCURRENT):
        namespace = 'grads_'+str(i)
        compute_gradients = self._optimizers[i].compute_gradients(cost, var_list=network_params)
        grad_vars = tf.get_collection(tf.GraphKeys.VARIABLES, namespace)
        assign_add_gradients_ops.append([tf.assign_add(v,g) for v,g in zip(grad_vars, [grad[0] for grad in compute_gradients])])
    self._assign_add_gradients_ops = assign_add_gradients_ops

    # Global optimizer (just to apply threads' gradients)
    apply_grads_optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    global_grads_and_vars = apply_grads_optimizer.compute_gradients(cost, var_list=network_params)
    self._global_grads_and_vars = global_grads_and_vars

    # Set up placeholder gradients
    placeholder_gradients = []
    for grad_var in global_grads_and_vars:
       placeholder_gradients.append((tf.placeholder('float', shape=grad_var[1].get_shape()) ,grad_var[1]))
    self._placeholder_gradients = placeholder_gradients

    # Op for applying the gradients
    apply_gradients = apply_grads_optimizer.apply_gradients(placeholder_gradients)
    self._async_apply_gradients = apply_gradients

    # Zero out gradients
    clear_gradients_ops = []
    for i in range(NUM_CONCURRENT):
        namespace = 'grads_'+str(i)
        grad_vars = tf.get_collection(tf.GraphKeys.VARIABLES, namespace)
        clear_gradients_ops.append([tf.assign(grad,np.zeros(grad.get_shape().as_list(), dtype=np.float32)) for grad in grad_vars])
    self._clear_gradients_ops = clear_gradients_ops

    # Global counter variable
    T = tf.Variable(0)
    self._increment_T = T.assign_add(1)
    self.T = T

    tf.initialize_all_variables().run()
    print("VARIABLES INITIALIZED")

  def _actor_learner_thread(self, i):
    """
    Main actor learner thread:
    - Initialize thread-specific environment
    - Run an infinite training loop, periodically
      sending asyncronous gradient updates to main model
    """
    # Initialize this thread's agent's environment
    environment = Environment(self._ale_io_lock)
    
    # Get initial game state
    state = environment.get_initial_state()

    epsilon = INITIAL_EXPLORATION

    # Main learning loop
    T_max = 1000000
    t = 0
    episode_reward = 0
    episode_max_q_value = 0
    episode_steps = 0
    while self._session.run(self.T) < T_max:
      # Get Q(s,a) values
      Q_s_a = self._session.run(self._network, feed_dict={self._state : state })

      # Take action a according to the e-greedy policy
      if epsilon > FINAL_EXPLORATION:
          epsilon -= (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME
      if self.rng.rand() < epsilon:
        action = self.rng.randint(0, environment.num_actions())
      else:
        action =  np.argmax(Q_s_a)

      # Execute the chosen action in the environment, observe new state and reward.
      (reward, new_state, terminal) = environment.execute(action)
      episode_reward += reward
      reward = np.clip(reward, -1, 1)

      # Compute y using target network
      target_Q_s_a = self._session.run(self._target_network, feed_dict={self._state : state })
      if terminal:
        y = reward
      else:
        y = reward + GAMMA * np.max(target_Q_s_a)

      # Build a one hot action vector
      actions_one_hot = np.zeros(environment.num_actions())
      actions_one_hot[action]=1.0

      # Calculate loss
      # loss = self._session.run(self._cost, feed_dict={self._state : state, self._a: [actions_one_hot], self._y: [y]})

      # Accumulate gradients
      self._session.run(self._assign_add_gradients_ops[i], feed_dict={self._state : state, self._a: [actions_one_hot], self._y: [y]})

      # Collect some stats (TODO: put in tensorflow summaries)
      episode_max_q_value += np.max(Q_s_a)
      episode_steps += 1

      # s = s'
      if not terminal:
        state = new_state
      else:
        episode_average_reward = episode_reward #/ float(episode_steps)
        episode_average_max_q_value = episode_max_q_value / float(episode_steps)
        print "Reward:", round(episode_average_reward, 3), "Max Q: ", round(episode_average_max_q_value,3)
        episode_reward = 0
        episode_max_q_value = 0 
        episode_steps = 0
        # Episode ended, so next state is the next episode's initial state
        state = environment.get_initial_state()    

      # T += 1
      self._session.run(self._increment_T)
      T = self._session.run(self.T)
      t += 1

      # Update the target network
      if T % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
        self._session.run(self._reset_target_network_params)

      # Perform asynchronous update
      if (t % NETWORK_UPDATE_FREQUENCY == 0) or terminal:
        grad_vals = self._session.run([grad for grad, _ in self._global_grads_and_vars], feed_dict={self._state : state, self._a: [actions_one_hot], self._y: [y]})

        # Build a variable name-to-gradient dictionary.
        var_to_grad = {}
        for j in range(len(grad_vals)):
            var = self._global_grads_and_vars[j][1]
            var_to_grad[var.name] = grad_vals[j]

        feed_dict = {}
        for j, (grad, var) in enumerate(self._global_grads_and_vars):
          feed_dict[self._placeholder_gradients[j][0]] = grad_vals[j]
        self._session.run(self._async_apply_gradients, feed_dict=feed_dict)

        self._session.run(self._clear_gradients_ops[i])

    print("=======================================done==============================")

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
      lock = threading.Lock()
      global_thread = GlobalThread(session, g, ale_io_lock, lock)
      global_thread.train()

if __name__ == "__main__":
  tf.app.run()

