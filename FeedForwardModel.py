from NeuralNetwork import NeuralNetwork
from tensorflow import keras


class FFM(NeuralNetwork):
    """ Feed forward neural network class"""

    def initialize_model(self):
        self.model.add(keras.layers.Dense(200, input_shape=(self.input_size,), activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(4))
        print(self.model.summary())
        self.model.compile(optimizer='adadelta', loss='mse')
