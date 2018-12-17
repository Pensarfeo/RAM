import tensorlayer as tl
import tensorflow as tf
import config

def previewNetwork(images):
        with tf.variable_scope('getPreview'):
            preview_imgs = tf.reshape(images, [tf.shape(images)[0], 28, 28, 1], name='reshape_layer_1')
            preview_imgs = tf.image.resize_images(preview_imgs, [config.win_size, config.win_size])
            preview_imgs = tf.reshape(preview_imgs, [tf.shape(images)[0], config.win_size * config.win_size * 1])
            
            preview_net = tl.layers.InputLayer(preview_imgs, name='preview_layer')
            preview_net = tl.layers.DenseLayer(preview_net, n_units = 128, name ='prev_fc_1')
            preview_net = tl.layers.BatchNormLayer(preview_net, name='prev_bn_1')
            preview_net = tf.nn.relu(preview_net.outputs, name='prev_relu_1')

            with tf.variable_scope('firstLocation'):
                preview_net = tl.layers.InputLayer(preview_imgs, name='preview_layer')
                preview_net = tl.layers.DenseLayer(preview_net, n_units = 128, name ='prev_fc_1')
                preview_net = tl.layers.BatchNormLayer(preview_net, name='prev_bn_1')
                preview_net = tf.nn.relu(preview_net.outputs, name='prev_relu_1')

                preview_net = tl.layers.InputLayer(preview_net, name='preview_layer_2')
                preview_net = tl.layers.DenseLayer(preview_net, n_units = config.loc_dim, name ='prev_fc_2')
                preview_net = tl.layers.BatchNormLayer(preview_net, name='prev_bn_2')
                preview_net = tf.nn.relu(preview_net.outputs, name='prev_relu_2')

                # Add random
                mean = tf.stop_gradient(tf.clip_by_value(preview_net, -1.0, 1.0))
                location = mean + tf.random_normal((tf.shape(state)[0], config.loc_dim), stddev=config.loc_std)
                location = tf.stop_gradient(location)

            with tf.variable_scope('firstGuess'):
                guess = tl.layers.InputLayer(preview_imgs)
                guess = tl.layers.DenseLayer(guess, n_units = config.num_classes, name='classification_guess_fc')
                guess = guess

        return location, mean, guess.outputs
