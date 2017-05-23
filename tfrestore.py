import tensorflow as tf

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow

import tfcifar2

# # Create a variable with a random value.
# weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
# # Create another variable with the same value as 'weights'.
# w2 = tf.Variable(weights.initialized_value(), name="w2")
# # Create another variable with twice the value of 'weights'
# w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")

#saver = tf.train.Saver({"my_v2": v2})

ckpt_path='../data/Pretrain/cifar10_train'
file_name= '../data/expresult/0521/exp03'
file_name_test='../data/Pretrain/cifar10_train/model.ckpt-186392'
tensor_name1='weights'
tensor_name2='bias'


# reader = pywrap_tensorflow.NewCheckpointReader(file_name_test)
# reader=tf.train.NewCheckpointReader(file_name_test)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print type(reader.get_tensor(key))
#     print(reader.get_tensor(key))


with tf.Session() as sess:
#    new_saver = tf.train.import_meta_graph('../data/Pretrain/cifar10_train/model.ckpt-186329.meta')
#    
#    new_saver.restore(sess, tf.train.latest_checkpoint('../data/Pretrain/cifar10_train'))
    
#    sess.run(tf.global_variables_initializer())
#    graph = tf.get_default_graph()
#    w1 = graph.get_tensor_by_name("conv1/weights:0")
#    w2 = graph.get_tensor_by_name("conv2/weights:0")
#    print w1,w2
#    print sess.run([w1,w2])
    tfcifar2.inference(None,None)
    
#    print_tensors_in_checkpoint_file(file_name=file_name, tensor_name=tensor_name1, all_tensors=True)
