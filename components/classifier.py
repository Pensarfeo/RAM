import tensorlayer as tl
import tensorflow as tf
import config

class ClassifierNetwork():
    def __init__(self):
        self.logits = []
        self.softmax = []

    def __call__(self, inputs):
        with tf.variable_scope('classifier', reuse = tf.AUTO_REUSE):
            net = tl.layers.InputLayer(inputs)
            net = tl.layers.DenseLayer(net, n_units = config.num_classes, name='classification_net_fc')
            net = net.outputs
            self.logits.append(net)
            self.softmax.append(tf.nn.softmax(net))
            self.net = net
        return 