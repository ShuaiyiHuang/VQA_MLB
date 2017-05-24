import tensorflow as tf
import re

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'
#tf.app.flags.DEFINE_boolean('use_fp16', False,
#                            """Train the model using fp16.""")

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

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
#    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype=tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
#  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  dtype=tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def restore_variables():
    return 

#return [batch_size,192]
def inference(images,batch_size):

#  
  print 'original input images:',images
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # img_shape = images.get_shape().as_list()
  # print 'img_shape',img_shape
  # batch_size=img_shape[0]
  # batch_size=images.get_shape()[0].value
  print 'batch_size',batch_size
  # conv1
  sess = tf.get_default_session()
  
  #if use pretrained cifar,resize images
  images=tf.image.resize_images(images,[24,24])
  print 'images in cifar2:',images
  new_saver = tf.train.import_meta_graph('../data/Pretrain/cifar10_train/model.ckpt-186329.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('../data/Pretrain/cifar10_train'))


  #achieved valid acccuray 0.699 in 4w ,run tfbaseline_fixq.py
  # file_name2_nouse='../data/expresult/0524/exp05/fixqexp05E34'
  # restore_path='../data/expresult/0524/exp05/fixqexp05E34'
  # new_saver = tf.train.import_meta_graph(file_name2_nouse+'.meta')
  # new_saver.restore(sess, restore_path)

  
  graph = tf.get_default_graph()
  w1 = graph.get_tensor_by_name("conv1/weights:0")
  b1=graph.get_tensor_by_name("conv1/biases:0")
  w2 = graph.get_tensor_by_name("conv2/weights:0")
  b2=graph.get_tensor_by_name("conv2/biases:0")
  w3=graph.get_tensor_by_name("local3/weights:0")
  b3=graph.get_tensor_by_name("local3/biases:0")
  w4=graph.get_tensor_by_name("local4/weights:0")
  b4=graph.get_tensor_by_name("local4/biases:0")
  print 'w1:',w1,'b1:',b1
  print 'w2',w2,'b2',b2
  print 'w3',w3,'b3',b3
  print 'w4',w4,'b4',b4
  print type(w1)
  #no need to print,it works
#  print 'in tfcifar2 restore:',sess.run([w1,b1,w2,b2,w3,b3,w4,b4])
#   print sess.run([w1,b1,w2,b2,w3,b3,w4,b4])
  
  with tf.variable_scope('conv1') as scope:
    
    conv = tf.nn.conv2d(images, w1, [1, 1, 1, 1], padding='SAME')

    pre_activation = tf.nn.bias_add(conv, b1)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    # _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2=[batch_size,16,16,64]
  with tf.variable_scope('conv2') as scope:
      
    conv = tf.nn.conv2d(norm1, w2, [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, b2)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    # _activation_summary(conv2)

  # norm2=[batch_size,16,16,64]
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2=[batch_size,8,8,64]
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  print 'conv1',conv1
  print 'norm1',norm1
  print 'pool1',pool1
  print 'conv2',conv2
  print 'norm2',norm2
  print 'pool2',pool2

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    #reshape=[batch_size,4096]
    reshape = tf.reshape(pool2, [batch_size, -1])
    print 'cifar reshape',reshape
    dim = reshape.get_shape()[1].value
    local3 = tf.nn.relu(tf.matmul(reshape, w3) + b3, name=scope.name)
    # _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:

    local4 = tf.nn.relu(tf.matmul(local3, w4) + b4, name=scope.name)
    # _activation_summary(local4)
  print 'cifar local 4',local4,type(local4)
#  res=sess.run(local4,feed_dict={})
#  print 'result return:',res[0:10,:]
  
  return local4