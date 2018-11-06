import tensorlayer as tl
import tensorflow as tf

from components.retina import Retina
from components.seq2seq import rnn_decoder
import config

def setUp(images_ph):
    retina = Retina(images_ph)
    
    # Construct core rnn network
    lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
    init_lstm_state = lstm_cell.zero_state(tf.shape(images_ph)[0], tf.float32)
    input_glimps_tensor_list = [retina.init_glimps_tensor]
    input_glimps_tensor_list.append([0] * config.num_glimpses)
    outputs, _ = rnn_decoder(input_glimps_tensor_list, init_lstm_state, lstm_cell, loop_function=retina.getNext)

    # Construct the classification network (action network?)
    action_net = tl.layers.InputLayer(outputs[-1]) 
    action_net = tl.layers.DenseLayer(action_net, n_units = config.num_classes, name='classification_net_fc')

    return [action_net.outputs, retina]