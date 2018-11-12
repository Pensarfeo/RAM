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
        self.images_ph = images_ph
        self.origin_coor_list = []
        self.sample_coor_list = []
        self.location_network = LocationNetwork()  
        self.glimps_network = GlimpsNetwork(images_ph)

    def firstGlimpse(self):
        init_location = tf.random_uniform((tf.shape(self.images_ph)[0], 2), minval=-1.0, maxval=1.0)
        self.origin_coor_list.append(init_location)
        output = self.glimps_network(init_location)

        return output

    def getNext(self, currentState, i):
        
        action_net = classifierNetwork(currentState)
        sample_coor, origin_coor = self.location_network(currentState, action_net.outputs)
        self.origin_coor_list.append(origin_coor)
        self.sample_coor_list.append(sample_coor)
        glimpse = self.glimps_network(sample_coor)
        return [glimpse, action_net]

def setUp(images_ph):    
    # Construct core rnn network
    retina = Retina(images_ph)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
    init_lstm_state = lstm_cell.zero_state(tf.shape(images_ph)[0], tf.float32)
    state = init_lstm_state
    input_glimps_tensor_list = []
    outputs = []
    inputs = []
    input_glimps_tensor_list.append([0] * config.num_glimpses)
    inputs.append(retina.firstGlimpse())

    with tf.variable_scope('coreNet'):

        for i in range(0, 6):
            REUSE = True if i>0 else None
            with variable_scope.variable_scope("rnn_decoder", reuse = REUSE):
                
                output, state = lstm_cell(inputs[-1], state)

                glimpse, action_net = retina.getNext(output, i)

                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                inputs.append(glimpse)
                outputs.append(output)


    return [action_net.outputs, retina]


