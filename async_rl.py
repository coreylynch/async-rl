
params = []
target_params = params.copy()
T = 0 # global counter
grads_and_vars = []

## Object that wraps an actor-learner's individual environment

# responsible for:
#  - Safely initializing and loading game
#  - Returning preprocessed game state to the actor-learner
#  - Excecuting actions on behalf of the actor-learner, returning the reward

class Environment(object):
    def __init__(self, ale_io_lock):
        self.ale = self.init_ale(ale_io_lock)

    def init_ale(self, ale_io_lock):
        ale_io_lock.acquire()
        ale = ALEInterface()
        ale.setInt('random_seed', 123)
        ale.loadROM('/Users/coreylynch/dev/atari_roms/breakout.bin')
        ale_io_lock.release()
        return ale
    
    def act(self, a):
        return self.ale.act(a)
    def get_legal_actions(self):
        return self.ale.getLegalActionSet()

    def rese


## TEMP ##
ale_io_lock = threading.Lock()

def actor_learner(thread_num):
    # Initialize thread-specific learning environment
    environment = Environment(ale_io_lock)
    legal_actions = environment.get_legal_actions()

    # TODO: read other repo to figure out how to interact w/ ale for DQN
    T_max = 100
    for episode in xrange(T_max):
      total_reward = 0
      while not ale.game_over():
        a = legal_actions[randrange(len(legal_actions))]
        reward = ale.act(a);
        total_reward += reward
      print 'Episode', episode, 'ended with score:', total_reward, 'from thread: ', thread_num
      ale.reset_game()

num_threads = 4
workers = []
for thread_num in xrange(num_threads):
  t = threading.Thread(target=actor_learner, args=[thread_num])
  t.start()
  workers.append(t)

## TEMP ## 


## Actor-learner thread callback
def learner():

    global optimizer 
    global network  

    t = 0 # thread step counter
    initial_state = get_initial_state()

    # take action according to e-greedy policy
    Q_s_a = network.forward(initial_state)
    action = argmax(Q_s_a)
    new_state, reward = emulator.excecute(action)
    if terminal:
        y = reward
    else:
        y = reward + gamma * max a Q(s_, a_, target_params)
    
    # accumulate gradients
    grads_and_vars += optimizer.compute_gradients(loss, <list of variables>)

    state = new_state
    T += 1
    t += 1
    if T % target_network_update_freq == 0:
        target_params = params.copy()
    if t % async_update_freq == 0:
        optimizer.apply_gradients(grads_and_vars)
        grads_and_vars = np.zeros(np.shape(grads_and_vars))



from random import randrange
from ale_python_interface import ALEInterface
import threading

io_lock = threading.Lock() # using a lock to avoid race condition on ale rom file I/O and init stuff

# Thread-safe way to have each actor-learner init ALE and load the game file
def init_ale():
    io_lock.acquire()
    ale = ALEInterface()
    ale.setInt('random_seed', 123)
    ale.loadROM('/Users/coreylynch/dev/atari_roms/breakout.bin')
    io_lock.release()
    return ale

def actor_learner(thread_num):
    # Initialize thread-specific learning environment
    ale = init_ale()
    legal_actions = ale.getLegalActionSet()

    T_max = 100
    for episode in xrange(T_max):
      total_reward = 0
      while not ale.game_over():
        a = legal_actions[randrange(len(legal_actions))]
        reward = ale.act(a);
        total_reward += reward
      print 'Episode', episode, 'ended with score:', total_reward, 'from thread: ', thread_num
      ale.reset_game()

num_threads = 4
workers = []
for thread_num in xrange(num_threads):
  t = threading.Thread(target=actor_learner, args=[thread_num])
  t.start()
  workers.append(t)


def network_trainable_variables():
  return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "network")

## Accum Gradients stuff ##
def build_graph():
  # ops to define deep q network graph
  # ops to define loss
  # op to compute gradient on loss
  # op to += gradients
  # op to zero out gradients
  optimizer = tf.train.AdamOptimizer(1e-4) 
  
  # Compute gradients using the optimizer and cost function, pass all trainable variables as argument
  getGrads = optimizer.compute_gradients(cost, var_list=tf.trainable_variables())
  
  # Assign add ( += operation ) new gradients to gradient variables
  grad_opList = [tf.assign_add(v,g) for v,g in zip(gradVarlist, [grad[0] for grad in getGrads])]
  
  # # Perform gradient averaging ( /= operation ) on accumulated gradients to calculate average grads.
  # div_opList = [tf.div(gradvar, batchSize) for gradvar in gradVarlist]
  
  # Apply gradients
  apply_grads = optimizer.apply_gradients(zip([tf.convert_to_tensor(grads) for grads in gradVarlist], tf.trainable_variables()))
  
  # Zero out gradients
  zero_out_grads = [tf.assign(grad,np.zeros(grad.get_shape().as_list(), dtype=np.float32)) for grad in gradVarlist]

# within each actor-learner thread
def accum_gradients(sess):
  sess.run(grad_opList, feed_dict={X: resizedImgs, Y: batch[1], p_keep_conv: 0.7, p_keep_hidden: 0.5}) # Iteratively accumulate your gradients
  
def async_update(sess, apply_grads):
  sess.run(apply_grads) # Apply gradients
  sess.run(zero_out_grads) # Zero out the gradient variables


## Target network stuff
import tensorflow as tf

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


# Define graph
with tf.variable_scope('network') as scope:
  l1_var_1 = _variable_on_cpu('weights1', (10),
                         tf.truncated_normal_initializer(stddev=0.01))
  l1_var_2 = _variable_on_cpu('weights2', (10),
                         tf.truncated_normal_initializer(stddev=0.01))
  l1_var_3 = _variable_on_cpu('weights3', (10),
                         tf.truncated_normal_initializer(stddev=0.01))

with tf.variable_scope('target_network') as scope:
  l2_var_1 = _variable_on_cpu('weights1', (10),
                        tf.truncated_normal_initializer(stddev=0.01))
  l2_var_2 = _variable_on_cpu('weights2', (10),
                        tf.truncated_normal_initializer(stddev=0.01))
  l2_var_3 = _variable_on_cpu('weights3', (10),
                         tf.truncated_normal_initializer(stddev=0.01))

# Build an initialization operation to run below.
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

with sess.as_default():
    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "network")

    second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "target_network")

    print first_train_vars
    print second_train_vars

    print "=====Before====="
    print first_train_vars[2].eval()
    print second_train_vars[2].eval()
    for i in range(len(first_train_vars)):
        second_train_vars[i] = second_train_vars[i].assign(first_train_vars[i])

    print "=====After====="
    print first_train_vars[2].eval()
    print second_train_vars[2].eval()
