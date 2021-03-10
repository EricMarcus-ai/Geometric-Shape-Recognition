import tensorflow as tf
from Defines import num_possible_shapes
import numpy as np
import os
import SegmentationCV


class NeuralNetwork:
    """ General neural network class"""

    def __init__(self, input_size, output_size=num_possible_shapes+4):
        self.model = tf.keras.Sequential()
        self.input_size = input_size
        self.output_size = output_size

    def save_model(self, model_name):
        self.model.save(model_name)

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def train_model(self, train_x, train_y, val_x, val_y, batch_size=512, epochs=100, model_name=''):

        if os.path.exists(model_name) and model_name:
            self.model.load_model(model_name)

        else:
            self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y),
                           verbose=2)

    def test_model(self, test_x, test_y):
        test_loss = self.model.evaluate(test_x, test_y, verbose=2)
        return test_loss

    def prediction(self, input_array):
        if input_array.shape == (self.input_size,):
            input_array = input_array.reshape(1, -1)
        predictions = self.model.predict(input_array)
        return predictions

    def multi_object_prediction(self, input_data):
        """
        Implements multi object recognition by first segmenting the image to single objects using non-ML computer vision techniques
        :param input_data: a collection of images containging multiple objects
        :return: list of prediction(s) of boxes for all inputted images
        """
        prediction_list = []
        for input_image in input_data:
            seg_images = SegmentationCV.segment(input_image)

            prediction_seg = []
            for i in range(len(seg_images)):
                prediction = self.prediction(seg_images[i])
                prediction_seg.append(prediction)

            prediction_seg = np.array([prediction_seg]).flatten()
            prediction_list.append(prediction_seg)

        return np.array(prediction_list)
