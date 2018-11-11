import tensorflow as tf
from components.glimpse import GlimpsNetwork
from components.location import LocationNetwork

class Retina(): #, _origin_coor_list, _sample_coor_list):
    def __init__(self, images_ph):
        self.origin_coor_list = []
        self.sample_coor_list = []
        self.location_network = LocationNetwork()  
        # init GlipseNetwork
        init_location = tf.random_uniform((tf.shape(images_ph)[0], 2), minval=-1.0, maxval=1.0)

        self.glimps_network = GlimpsNetwork(images_ph)
        self.init_glimps_tensor = self.glimps_network(init_location)

    def getNext(self, output, i): 
        sample_coor, origin_coor = self.location_network(output)
        self.origin_coor_list.append(origin_coor)
        self.sample_coor_list.append(sample_coor)
        return self.glimps_network(sample_coor)