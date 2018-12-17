import tensorlayer as tl
import tensorflow as tf

from components.seq2seq import rnn_decoder

from components.glimpse import GlimpsNetwork
from components.location import LocationNetwork
from components.classifier import ClassifierNetwork
from components.preview import previewNetwork



import config

class StateTracker():
    def __init__(self, stateSize):
        with tf.variable_scope('stator', reuse = None):
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
            self.state = self.lstm_cell.zero_state(stateSize, tf.float32)

    def __call__(self, glimpse):
        self.output, self.state = self.lstm_cell(glimpse, self.state)
        return self.output

class CoreNetwork():
    def __init__(self, images_ph):
        self.images_ph = images_ph
        # init core networks
        self.location_network = LocationNetwork()  
        self.glimpse = GlimpsNetwork(images_ph)
        self.classifier = ClassifierNetwork()
        self.stateTracker = StateTracker(tf.shape(images_ph)[0])

        # memory
        self.locations = []
        self.means = []
        self.guesses = []

        # first pass though the network
        self.firstGlimpse()
        self.input = None

    def firstGlimpse(self):
        location, mean, guess = previewNetwork(self.images_ph)
        
        self.locations.append(location)
        self.means.append(mean)
        self.guesses.append(guess)

    def __call__(self):
        glimpse = self.glimpse(self.locations[-1], self.guesses[-1])
        currentState = self.stateTracker(glimpse)
        self.guesses += self.classifier(currentState)
        location, mean = self.location_network(currentState, self.guesses[-1])
        self.locations.append(location)
        self.means.append(mean)

def setUp(images_ph):    
    # Construct core rnn network
    with tf.variable_scope('coreNetwork', reuse = tf.AUTO_REUSE):
        coreNetwork = CoreNetwork(images_ph)
        for i in range(0, 6):
            with tf.variable_scope('rnn'):
                coreNetwork()

    return coreNetwork


