import tensorflow as tf
import os


class NeuralNetwork:

    def __init__(self):
        self.model = tf.keras.Sequential()

    def train_model(self, combined_train, model_name, force_retrain=False):

        if os.path.exists(model_name) and force_retrain is False:
            self.model.load_model(model_name)

        else:
            self.model.fit(combined_train, batch_size=2048, epochs=2)
