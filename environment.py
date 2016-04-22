import tensorflow as tf
from ale_python_interface import ALEInterface
from skimage.transform import resize
import numpy as np
from collections import deque

ACTIONS = 4
NUM_CONCURRENT = 1
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
    self.max_start_nullops = NOOP_MAX
    self.rng = np.random.RandomState(123456)
    self.resized_width = RESIZED_WIDTH
    self.resized_height = RESIZED_HEIGHT
    self.agent_history_length = AGENT_HISTORY_LENGTH

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

    new_state = np.reshape(new_state, (1, 84, 84, 4))
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
    resized = resize(max_image, (self.resized_width, self.resized_height))
    return resized