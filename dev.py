
import tensorflow as tf
from random import randrange
from ale_python_interface import ALEInterface
import threading

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

class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, session, graph):
    self._session = session
    self._graph = graph
    self.build_graph()

  def build_graph(self):
    with tf.variable_scope('local1') as scope:
      l1_var_1 = _variable_on_cpu('weights1', (10),
                             tf.truncated_normal_initializer(stddev=0.01))
      l1_var_2 = _variable_on_cpu('weights2', (10),
                             tf.truncated_normal_initializer(stddev=0.01))
      l1_var_3 = _variable_on_cpu('weights3', (10),
                             tf.truncated_normal_initializer(stddev=0.01))
    
    with tf.variable_scope('local2') as scope:
      l2_var_1 = _variable_on_cpu('weights1', (10),
                            tf.truncated_normal_initializer(stddev=0.01))
      l2_var_2 = _variable_on_cpu('weights2', (10),
                            tf.truncated_normal_initializer(stddev=0.01))
      l2_var_3 = _variable_on_cpu('weights3', (10),
                             tf.truncated_normal_initializer(stddev=0.01))

    
    network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "local1")
    target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        "local2")

    
    print network_params
    print target_network_params
    self._copy = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

    tf.initialize_all_variables().run()
    print("VARIABLES INITIALIZED")


  def _train_thread_body(self):
  	print "running"
  	self._session.run(self._copy)

  def train(self):
    workers = []
    for _ in xrange(5):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

# Thread-safe way to have each actor-learner init ALE and load the game file
def init_ale():
    io_lock.acquire()
    ale = ALEInterface()
    ale.setInt('random_seed', 123)
    ale.loadROM('/Users/coreylynch/dev/atari_roms/breakout.bin')
    io_lock.release()
    return ale

def main(_):
  
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(session, g)
      model.train()  # Process one epoch
      network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "local1")
      target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "local2")

      print network_params[2].eval()
      print target_network_params[2].eval()


if __name__ == "__main__":
  tf.app.run()

