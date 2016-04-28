import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque

class ActorLearner(object):
    """
    Agent responsible for:
    - Processing raw observations from the gym environment.
    - Maintaining its own state over observations
      (e.g. a set of 4 previous processed frames).
    - Taking actions according to e-greedy policy.
    - Computing gradients w.r.t. global network params and
      updating the global network asyncronously.

    """
    def __init__(self, action_space, final_epsilon):
        self.action_space = action_space
        self.starting_epsilon = 1
        self.final_epsilon = final_epsilon

        self.state_buffer = deque()
        self.agent_history_length = 4
        self.resized_width = 84
        self.resized_height = 84

    def act(self, observation, reward, done):
        """
        Choose action based on current state and e-greedy policy
        """
        state = self._get_state(observation)
        print np.shape(state)

        # For now, return random action
        return self.action_space.sample()

    def build_initial_state(self, observation):
        """
        Fill the state buffer with the initial observation
        """
        processed_frame = self._get_preprocessed_frame(observation)
        self.state_buffer = deque() # clear the state buffer
        for _ in xrange(self.agent_history_length-1):
            self.state_buffer.append(processed_frame)

    def _get_state(self, observation):
        # Preprocess the current frame
        processed_frame = self._get_preprocessed_frame(observation)

        # Build state by concatenating current processed frame with previous
        # (agent_history_length-1) in the state buffer.
        state = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        previous_frames = np.array(self.state_buffer)
        state[:self.agent_history_length-1, ...] = previous_frames
        state[self.agent_history_length-1] = processed_frame
        
        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(processed_frame)
        return [state]

    def _get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        Note: Cannot take the max over the current and previous frame
        as they do in the paper, because gym skips a random amount of
        frames on each .act. Maybe try a workaround if performance
        really suffers.
        """
        return resize(rgb2gray(observation), (self.resized_width, self.resized_height))

