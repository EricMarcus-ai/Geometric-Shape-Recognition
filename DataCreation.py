import numpy as np
from Defines import background_size, label_size, num_boxes
import random
import Utils


class ShapeCreator:
    """ this class will create the data sets containing the shapes """

    def __init__(self):
        self.back_size = background_size  # how large the background will be, on which the shapes are created
        self.label_size = label_size  # how many entries are there in one label
        self.num_boxes = num_boxes  # how many boxes will be drawn

    def box_creator(self, flatten=True):
        # this method creates one box object with top left location at x,y
        # the w (weight) increases to the right and h (height) downwards
        box_array = np.zeros((self.back_size, self.back_size))  # creating background
        label_array = np.zeros(self.num_boxes * self.label_size)

        for i in range(self.num_boxes):
            x = random.randint(0, self.back_size-2)  # no x coords on the boundary - that would yield empty boxes
            y = random.randint(0, self.back_size-2)
            w = random.randint(1, (self.back_size-1)-x)  # width at least 1
            h = random.randint(1, (self.back_size-1)-y)

            box_array[y:y+h, x:x+w] = 1
            label_array[label_size * i:label_size * i+label_size] = [x, y, w, h]

        # note that we normalize the x,y,w,h by the background size in the return below
        if flatten:
            return box_array.flatten(), label_array / self.back_size
        else:
            return [box_array, [label_array] / self.back_size]

    def box_dataset_creator(self, num_imgs, filename=''):
        # This method always returns a non-flattened box_list and (if passed) saves a flattened version in csv
        box_list = np.empty((num_imgs, self.back_size * self.back_size))
        label_list = np.empty((num_imgs, self.label_size * self.num_boxes))
        for i in range(num_imgs):
            local_box, local_label = self.box_creator()
            box_list[i] = local_box
            label_list[i] = local_label

        # below I normalize to mean 0 and std 1, although this doesn't matter much for the eventual training
        # probably because feature normalization is mostly necessary when features have widely varying sizes
        box_list = (box_list - np.mean(box_list)) / np.std(box_list)

        if filename:  # if saved, we save the box_list flattened in csv format
            np.savetxt(filename + '.csv', box_list, delimiter=',')
            np.savetxt(filename + '_labels' + '.csv', label_list, delimiter=',')

        return box_list, label_list

    @staticmethod
    def box_dataset_loader(file_path):
        # This method loads a dataset saved in csv format
        box_list = np.genfromtxt(file_path + '.csv', delimiter=',')
        label_list = np.genfromtxt(file_path + '_labels' + '.csv', delimiter=',')

        return box_list, label_list
