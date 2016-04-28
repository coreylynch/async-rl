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
    def __init__(self, action_space, session, graph_ops,
                 final_epsilon, final_epsilon_frame):
        self.session = session # tf session
        self.graph_ops = graph_ops # dict of tf graph ops

        self.action_space = action_space
        self.starting_epsilon = 1
        self.epsilon = self.starting_epsilon
        self.final_epsilon = final_epsilon
        self.final_epsilon_frame = final_epsilon_frame

        self.state_buffer = deque()
        self.agent_history_length = 4
        self.resized_width = 84
        self.resized_height = 84

        self.rng = np.random.RandomState(np.random.randint(10000))

    def choose_action(self, observation, reward, done):
        """
        Choose action based on current state and e-greedy policy
        Returns action index and state.
        """
        # Get current state
        state = self.get_state(observation)
        
        # Get Q(s,a) values
        Q_s_a = self.session.run(self.graph_ops['network'], feed_dict={self.graph_ops['state']: state})[0]

        # E-greedy action selection
        a = self._select_action_e_greedy(Q_s_a)

        # Scale down epsilon
        self._decay_epsilon()

        # Collect max q value
        max_q_value = np.max(Q_s_a)

        return (a, state, max_q_value)

    def build_initial_state(self, observation):
        """
        Fill the state buffer with the initial observation
        """
        processed_frame = self._get_preprocessed_frame(observation)
        self.state_buffer = deque() # clear the state buffer
        for _ in xrange(self.agent_history_length-1):
            self.state_buffer.append(processed_frame)

    def get_state(self, observation):
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

    def _select_action_e_greedy(self, Q_s_a):
        if self.rng.rand() < self.epsilon:
            a = self.rng.randint(0, self.action_space.n)
        else:
            a = np.argmax(Q_s_a)
        return a

    def _decay_epsilon(self):
        """
        Scale down epsilon based on global step count T
        """
        T = self.session.run(self.graph_ops["T"])
        if self.epsilon > self.final_epsilon:
          self.epsilon = self.starting_epsilon - ((self.starting_epsilon-self.final_epsilon) * float(T)/self.final_epsilon_frame)

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

