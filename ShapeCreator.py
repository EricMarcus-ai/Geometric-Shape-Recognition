import numpy as np
import random
from Defines import num_possible_shapes
import cv2


class ShapeCreator:

    def __init__(self, background_size=8, num_objects=1, bbox_size=4):
        self.back_size = background_size
        self.bbox_size = bbox_size
        self.num_objects = num_objects

    def make_object(self, image, num_objects, object_type):
        """
        Places num_objects objects on image, if object_type is not inputted they are chosen randomly
        :param image: the image upon which objects are placed
        :param num_objects: how many objects will be built
        :param object_type: can be 0 for box,1 for cirlce,2 for triangle or None, if None => will be chosen randomly in (0,num_possible_shapes-1)
        :return: returns the inputted image with objects placed upon it
        """
        label = []
        for i in range(num_objects):
            object_type = self.object_type_chooser(object_type)
            label.append(self.one_hot_encoder(object_type))

            if object_type == 0:
                image, bbox = self.box_creator(image)
                label.append(bbox)
            if object_type == 1:
                image, bbox = self.circle_creator(image)
                label.append(bbox)
            if object_type == 2:
                image, bbox = self.triangle_creator(image)
                label.append(bbox)

        label = np.concatenate(label)
        return image, label

    def dataset_creator(self, num_imgs, object_type=None, filename='', flatten=True):
        """
        Creates datasets using the make_object method
        :param num_imgs: number of images to be created
        :param object_type: specify to 0,1,2 to get only boxes, circles or triangles, None => randomly distributed dataset
        :param filename: set filename if the dataset is to be saved
        :param flatten: set true for FeedForwardModel and False for ConvolutionalModel (formats to input for the models)
        :return: yields (image_list, label_list) dataset
        """
        image_list = np.zeros((num_imgs, self.back_size, self.back_size))
        label_list = np.empty((num_imgs, (num_possible_shapes + self.bbox_size) * self.num_objects))

        for i in range(num_imgs):
            image_list[i], label_list[i] = self.make_object(image_list[i], self.num_objects, object_type)

        if filename:
            np.savetxt(filename + '.csv', image_list, delimiter=',')
            np.savetxt(filename + '_labels' + '.csv', label_list, delimiter=',')

        if flatten:
            return image_list.reshape(-1, self.back_size ** 2), label_list
        else:
            return image_list, label_list

    def box_creator(self, image):
        """ creates one box object upon image with top left location at x,y
        and the w (width) increases to the right and h (height) downwards """

        x = random.randint(0, self.back_size-2)  # no x coords on the right boundary - that would yield empty boxes
        y = random.randint(0, self.back_size-2)
        w = random.randint(1, (self.back_size-1)-x)  # width at least 1
        h = random.randint(1, (self.back_size-1)-y)

        image[y:y+h, x:x+w] = 1
        bbox = np.array([x, y, w, h])

        return image, bbox / self.back_size  # normalize the x,y,w,h by the background size

    def circle_creator(self, image):
        max_coordinate = self.back_size - 1

        x_center = random.randint(1, max_coordinate-1)  # center shouldn't be on the boundary of the image
        y_center = random.randint(1, max_coordinate-1)
        radius_max = min(x_center, max_coordinate - x_center, y_center, max_coordinate - y_center)  # make sure it doesn't go out of bounds
        radius = random.randint(1, radius_max)

        bbox = np.array([x_center-radius, y_center-radius, 2 * radius + 1, 2 * radius + 1])  # +1 for w and h due to thickness added by cv2

        image = cv2.circle(image, center=(x_center, y_center), radius=radius, color=1, thickness=-1)
        return image, bbox / self.back_size

    def triangle_creator(self, image):
        min_coordinate = 1  # add padding to allow for thickness of polygon lines
        max_coordinate = self.back_size - 2

        point1 = [random.randint(0, max_coordinate-1), random.randint(1, max_coordinate-1)]
        point2 = [random.randint(point1[0]+1, max_coordinate), random.randint(point1[1]+1, max_coordinate)]
        point3 = [point2[0], point1[1]]
        vertices = np.array([point1, point2, point3]).reshape((-1, 1, 2))

        image = cv2.polylines(image, [vertices], isClosed=True, color=1, thickness=1)
        image = cv2.fillPoly(image, [vertices], color=1)

        x = point1[0]
        y = point1[1]
        w = point2[0] - point1[0] + 1  # add 1 due to thickness added by cv2
        h = point2[1] - point1[1] + 1
        bbox = np.array([x, y, w, h])

        return image, bbox / self.back_size

    @staticmethod
    def dataset_loader(file_path):
        box_list = np.genfromtxt(file_path + '.csv', delimiter=',')
        label_list = np.genfromtxt(file_path + '_labels' + '.csv', delimiter=',')

        return box_list, label_list

    @staticmethod
    def object_type_chooser(object_type):
        if object_type is None:
            object_type = random.randint(0, num_possible_shapes-1)
        return object_type

    @staticmethod
    def one_hot_encoder(object_type):
        one_hot_label = np.zeros(num_possible_shapes)
        one_hot_label[object_type] = 1
        return one_hot_label
