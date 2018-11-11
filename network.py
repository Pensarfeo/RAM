import tensorlayer as tl
import tensorflow as tf

from tensorflow.python.ops import variable_scope

from components.retina import Retina
from components.seq2seq import rnn_decoder
from components.classifier import classifierNetwork
from components.glimpse import GlimpsNetwork
from components.location import LocationNetwork

import config


class Retina():
    def __init__(self, images_ph):
        self.origin_coor_list = []
        self.sample_coor_list = []
        self.location_network = LocationNetwork()  
        self.glimps_network = GlimpsNetwork(images_ph)

    def getNext(self, currentState, i):
        action_net = classifierNetwork(currentState)
        sample_coor, origin_coor = self.location_network(currentState, action_net.outputs)
        self.origin_coor_list.append(origin_coor)
        self.sample_coor_list.append(sample_coor)
        glimpse = self.glimps_network(sample_coor)

        return [glimpse, action_net]

def setUp(images_ph):    
    # Construct core rnn network
    with tf.variable_scope('coreNet'):

        retina = Retina(images_ph)

        with variable_scope.variable_scope("rnn_decoder"):

            input_glimps_tensor_list = []
            outputs = []

            input_glimps_tensor_list.append([0] * config.num_glimpses)

            with tf.variable_scope('lstm_cell'):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
                init_lstm_state = lstm_cell.zero_state(tf.shape(images_ph)[0], tf.float32)

            state = init_lstm_state
            outputs = []

            for i, inp in enumerate(input_glimps_tensor_list):
                output, state = lstm_cell(inp, state)

                with variable_scope.variable_scope("loop_function", reuse=None):    # original set as True
                    inp = retina.getNext(output, i)

                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                
                outputs.append(output)


    return [action_net.outputs, retina]


