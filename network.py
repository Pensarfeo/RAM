import tensorlayer as tl
import tensorflow as tf

from components.retina import Retina
from components.seq2seq import rnn_decoder

import config

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


    with tf.variable_scope('coreNetwork', reuse = tf.AUTO_REUSE):
        for i in range(0, 6):
            with tf.variable_scope('rnn'):

                output, state = lstm_cell(inputs[-1], state)

                glimpse = retina.getNext(output)

                inputs.append(glimpse)
                outputs.append(output)

    return retina


