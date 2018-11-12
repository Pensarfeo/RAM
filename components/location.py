import tensorlayer as tl
import tensorflow as tf
import config

class LocationNetwork(object):
    def __call__(self, state_tensor, guess):
        with tf.variable_scope(tf.get_variable_scope()) as vs:
            # Network structure
            self.location_net = tl.layers.InputLayer(state_tensor)
            self.location_net = tl.layers.DenseLayer(self.location_net, n_units = config.loc_dim, name='location_net_fc1')

            # Add random
            self.mean = tf.stop_gradient(tf.clip_by_value(self.location_net.outputs, -1.0, 1.0))
            self.location = self.mean + tf.random_normal((tf.shape(state_tensor)[0], config.loc_dim), stddev=config.loc_std)
            self.location = tf.stop_gradient(self.location)
            return self.location, self.mean