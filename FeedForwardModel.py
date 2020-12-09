from NeuralNetwork import NeuralNetwork
from Defines import background_size
from tensorflow import keras


class FeedForwardNN(NeuralNetwork):

    def __init__(self):
        self.input_size = background_size * background_size

    def initialize_model(self):
        self.model.add(keras.layers.Dense(100, input_shape=self.input_size))
