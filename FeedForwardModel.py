from NeuralNetwork import NeuralNetwork
from tensorflow import keras


class FFM(NeuralNetwork):
    """ Feed forward neural network class"""

    def initialize_model(self):
        self.model.add(keras.layers.Dense(400, input_shape=(self.input_size,), activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(100, input_shape=(self.input_size,), activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(self.output_size))
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mse')
