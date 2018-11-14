import tensorflow as tf
from components.glimpse import GlimpsNetwork
from components.location import LocationNetwork
from components.classifier import classifierNetwork

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

    def getNext(self, currentState, i, glimpseScope = None):
        action_net = classifierNetwork(currentState)
        sample_coor, origin_coor = self.location_network(currentState, action_net.outputs)
        self.origin_coor_list.append(origin_coor)
        self.sample_coor_list.append(sample_coor)
        glimpse = self.glimps_network(sample_coor)
        return [glimpse, action_net]