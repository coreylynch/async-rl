
import tensorflow as tf
from random import randrange
from ale_python_interface import ALEInterface
import threading
from skimage.transform import resize
import numpy as np
from collections import deque

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


class Environment(object):
  """
  Object that wraps each actor-learner's individual game environment.
  Responsible for:
   - Safely initializing and loading game
   - Returning preprocessed game state to the actor-learner
   - Excecuting actions on behalf of the actor-learner, returning the reward
  """
  def __init__(self, ale_io_lock):
    self.ale = self.init_ale(ale_io_lock)
    self.frames_to_skip = 4
    self.min_action_set = self.ale.getMinimalActionSet()
    self.buffer_length = 2
    self.width, self.height = self.ale.getScreenDims()
    self.screen_buffer = np.empty((self.buffer_length,
                                   self.height, self.width),
                                  dtype=np.uint8)
    self.state_buffer = deque()

    self.index = 0
    self.max_start_nullops = 30
    self.rng = np.random.RandomState(123456)
    self.resized_width = 84
    self.resized_height = 84
    self.agent_history_length = 4

  def init_ale(self, ale_io_lock):
    ale_io_lock.acquire()
    ale = ALEInterface()
    ale.setInt('random_seed', 123)
    ale.loadROM('/Users/coreylynch/dev/atari_roms/breakout.bin')
    ale_io_lock.release()
    return ale

  def num_actions(self):
    return len(self.min_action_set)

  def _repeat_action(self, action):
    """
    From Mnih et al., "...the agent sees and selects actions on
    every kth frame instead of every frame, and its last action
    is repeated on skipped frames." 
    This repeats the chosen action the appopriate number of times 
    and returns the summed reward. 
    """
    reward = 0
    for _ in range(self.frames_to_skip):
        reward += self._act(action)
    return reward

  def _act(self, action):
    """
    Perform the indicated action for a single frame, return the
    resulting reward and store the resulting screen image in the
    buffer.
    """
    reward = self.ale.act(action)
    self.ale.getScreenGrayscale(self.screen_buffer[self.index, ...])
    self.index = (0 if self.index == 1 else 1)
    return reward

  def _init_new_episode(self):
    """ Resets the game if needed, performs enough null
    actions to ensure that the screen buffer is ready and optionally
    performs a randomly determined number of null action to randomize
    the initial game state."""
    self.ale.reset_game()
    self.start_lives = self.ale.lives()
    self.state_buffer = deque() # clear the state buffer
    if self.max_start_nullops > 0:
        random_actions = self.rng.randint(0, self.max_start_nullops+1)
        for _ in range(random_actions):
            self._act(0) # Null action

    # Make sure the screen buffer and state buffer is filled at the beginning of
    # each episode...
    for i in range(self.agent_history_length-1):
        self._act(0) # Null action
        self.state_buffer.append(self.get_preprocessed_frame())

  def get_initial_state(self):
    """ Inits a new episode, returns the current state (np array with agent_history_length
    most recent frames."""
    self._init_new_episode()
    return self.get_new_state()

  def _did_episode_end(self):
    """Returns a boolean indicating whether or not
    the game ended"""
    loss_of_life = self.ale.lives() < self.start_lives
    terminal = self.ale.game_over() or loss_of_life
    return terminal

  def get_new_state(self):
    """
    State is the agent_history_length recent frames presented to the agent. Mnih et al. set
    this to 4. This method creates a numpy array of size [agent_history_length, resized_height, 
    resized_width]. First, get the current preprocessed frame. Then grab the previous 
    agent_history_length-1 frames from the state_buffer (a deque) and put all agent_history_length
    into a state
    """
    # Get the current preprocessed frame
    current_frame = self.get_preprocessed_frame()
    
    # Get the most recent agent_history_length-1 frames from the self.state_buffer deque
    # Concatenate w/ current frame to get full current state: numpy array
    # of shape [agent_history_length, resized_height, resized_width]
    previous_frames = np.array(self.state_buffer)
    new_state = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
    new_state[:self.agent_history_length-1, ...] = previous_frames
    new_state[self.agent_history_length-1] = current_frame

    # Pop the oldest frame, add the current frame to the queue
    self.state_buffer.popleft()
    self.state_buffer.append(current_frame)

    return new_state

  def execute(self, action):
    """Executes an action in the environment on behalf of
    the agent. 
    Returns: - The instantaneous reward
             - The resulting preprocessed game state 
             - A boolean indicating whether or 
               not the executed action ended the game."""
    reward = self._repeat_action(self.min_action_set[action])
    new_state = self.get_new_state() # Rotates the state buffer by one
    terminal = self._did_episode_end()
    return (reward, new_state, terminal)

  def get_preprocessed_frame(self):
    """ 
    See Methods->Preprocessing in Mnih et al.
    1) Get image grayscale
    2) Take the maximum value for each pixel color
       value over the frame being encoded and the previous frame
       (to avoid flickering issues)
    3) Rescale image
    """
    max_image = np.maximum(self.screen_buffer[self.index, ...],
                           self.screen_buffer[(0 if self.index == 1 else 1), ...])
    return resize(max_image, (self.resized_width, self.resized_height))

class GlobalThread(object):
  """
  Main global thread responsible for:
    - Initializing shared global network and target network parameters
    - Kicking off actor-learner threads
  """
  def __init__(self, session, graph, ale_io_lock):
    self._session = session
    self._graph = graph
    self._ale_io_lock = ale_io_lock
    self.build_graph()
    self.rng = np.random.RandomState(123456)

  def build_graph(self):
    """ Placeholder for the function that builds up the model 
    """
    # Initialize online network and target network.
    with tf.variable_scope('network_params') as scope:
      l1_var_1 = _variable_on_cpu('weights1', (10),
                             tf.truncated_normal_initializer(stddev=0.01))
      l1_var_2 = _variable_on_cpu('weights2', (10),
                             tf.truncated_normal_initializer(stddev=0.01))
      l1_var_3 = _variable_on_cpu('weights3', (10),
                             tf.truncated_normal_initializer(stddev=0.01))
    
    with tf.variable_scope('target_network_params') as scope:
      l2_var_1 = _variable_on_cpu('weights1', (10),
                            tf.truncated_normal_initializer(stddev=0.01))
      l2_var_2 = _variable_on_cpu('weights2', (10),
                            tf.truncated_normal_initializer(stddev=0.01))
      l2_var_3 = _variable_on_cpu('weights3', (10),
                             tf.truncated_normal_initializer(stddev=0.01))

    # Op for periodically updating target network with online network weights
    network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "network_params")
    target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "target_network_params")
    self._reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

    # Global variables
    self._epoch = tf.Variable(0)

    tf.initialize_all_variables().run()
    print("VARIABLES INITIALIZED")

  def _increment(self, var):
    """ var += 1 """
    self._session.run(var.assign_add(1))

  def _actor_learner_thread(self):
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

    # initialize thread specific optimizer for computing gradients

    # Main learning loop
    T = 0
    T_max = 100
    while T < T_max:
      # Get predictions
      # Q_s_a = self.network.forward(state)

      # Take action a according to the e-greedy policy
      # TODO: thread-specific exploration policy
      # (periodically sample epsilon from some dist.)
      epsilon = 1
      if self.rng.rand() < epsilon:
        action = self.rng.randint(0, environment.num_actions())
      else:
        action =  np.argmax(Q_s_a)

      # Execute the chosen action
      (reward, new_state, terminal) = environment.execute(action)
      if reward > 0:
        print np.shape(new_state)
        print "action: ", action
        print "reward: ", reward
        print "terminal: ", terminal

      # Compute y using target network

      # Accumulate gradients

      if not terminal:
        state = new_state
      else:
        print ("Episode ended")
        # Episode ended, so next state is the next episode's initial state
        state = environment.get_initial_state()

      T += 1
      # if T % opt.target_frequency == 0:
      #   self.update_target_network()
      # if t % opt.async_update_frequency == 0:
      #   self.async_update()
      #   self.clear_gradients()

    self._session.run(self._reset_target_network_params) # Update the target network periodically
    self._increment(self._epoch)

  def train(self):
    """
    Creates and kicks off num_concurrent actor-learner threads
    """
    workers = []
    num_concurrent = 4
    for _ in xrange(num_concurrent):
      t = threading.Thread(target=self._actor_learner_thread)
      t.start()
      workers.append(t)
    for t in workers:
      t.join()

  def init_ale(self):
    """
    Thread-safe way to have each actor-learner init ALE and load the game file
    """
    self._ale_io_lock.acquire()
    ale = ALEInterface()
    ale.setInt('random_seed', 123)
    ale.loadROM('/Users/coreylynch/dev/atari_roms/breakout.bin')
    self._ale_io_lock.release()
    return ale


def main(_):
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      ale_io_lock = threading.Lock() # using a lock to avoid race condition on ALE init
      model = GlobalThread(session, g, ale_io_lock)
      model.train()


      # Just some temporary stuff to make sure target network is being copied correctly.
      # Delete later.
      network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "network_params")
      target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "target_network_params")

      print network_params[2].eval()
      print target_network_params[2].eval()


if __name__ == "__main__":
  tf.app.run()

