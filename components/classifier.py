import tensorlayer as tl
import tensorflow as tf
import config

def classifierNetwork(inputs):
    with tf.variable_scope('classifier', reuse = tf.AUTO_REUSE):
        action_net = tl.layers.InputLayer(inputs)
        action_net = tl.layers.DenseLayer(action_net, n_units = config.num_classes, name='classification_net_fc')

    return action_net