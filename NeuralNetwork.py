import tensorflow as tf
import os
from Defines import background_size


class NeuralNetwork:
    """ General neural network class"""

    def __init__(self):
        self.model = tf.keras.Sequential()
        self.input_size = background_size * background_size

    def save_model(self, model_name):
        self.model.save(model_name)

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def train_model(self, train_x, train_y,  model_name, batch_size=2048, epochs=2, force_retrain=False):

        if os.path.exists(model_name) and force_retrain is False:
            self.model.load_model(model_name)

        else:
            self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

    def test_model(self, test_x, test_y):
        test_loss = self.model.evaluate(test_x, test_y, verbose=2)
        return test_loss

    def prediction(self, input_array):
        if input_array.shape == (64,):
            input_array = input_array.reshape(1, -1)
        predictions = self.model.predict(input_array)
        return predictions
