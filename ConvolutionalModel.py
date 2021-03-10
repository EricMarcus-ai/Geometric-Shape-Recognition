from NeuralNetwork import NeuralNetwork
from tensorflow import keras


class CNN(NeuralNetwork):

    def initialize_model(self):
        self.model.add(keras.layers.Input(self.input_size))
        self.model.add(keras.layers.ZeroPadding2D((1, 1), data_format='channels_first'))

        self.model.add(keras.layers.Conv2D(32, (11, 11), strides=(3, 3), data_format='channels_first'))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_first'))

        self.model.add(keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format='channels_first'))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), data_format='channels_first'))

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(300))
        self.model.add(keras.layers.Dense(self.output_size))
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mse')
