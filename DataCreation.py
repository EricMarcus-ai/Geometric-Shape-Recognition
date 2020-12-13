import numpy as np
from Defines import background_size, label_size
import random
import Utils


class ShapeCreator:
    """ this class will create the data sets containing the shapes """

    def __init__(self):
        self.back_size = background_size  # how large the background will be, on which the shapes are created
        self.label_size = label_size  # how many entries are there in one label

    def box_creator(self, flatten=True):
        # this method creates one box object with top left location at x,y
        # the w (weight) increases to the right and h (height) downwards
        box_array = np.zeros((self.back_size, self.back_size))  # creating background

        x = random.randint(0, self.back_size-2)  # no x coords on the boundary - that would yield empty boxes
        y = random.randint(0, self.back_size-2)
        w = random.randint(1, (self.back_size-1)-x)  # width at least 1
        h = random.randint(1, (self.back_size-1)-y)

        box_array[y:y+h, x:x+w] = 1

        # note that we normalize the x,y,w,h by the background size in the return below
        if flatten:
            return box_array.flatten(), np.array([x, y, w, h]) / self.back_size
        else:
            return [box_array, np.array([[x, y, w, h]]) / self.back_size]

    def box_dataset_creator(self, num_imgs, filename, saved=False):
        # TODO: saved is broken for now, since I am adapting the output to NN input and haven't updated the save
        # This method always returns a non-flattened box_list and (if passed) saves a flattened version in csv
        box_list = np.empty((num_imgs, background_size * background_size))
        label_list = np.empty((num_imgs, label_size))
        for i in range(num_imgs):
            local_box, local_label = self.box_creator()
            box_list[i] = local_box
            label_list[i] = local_label

        if saved:  # if saved, we save the box_list flattened in csv format
            save_list = []
            for i in range(len(box_list)):
                save_list.append(np.hstack(box_list[i]))
            np.savetxt(filename, save_list, delimiter=',')

        # below I normalize to mean 0 and std 1, although this doesn't matter much for the eventual training
        # probably because feature normalization is mostly necessary when features have widely varying sizes
        box_list = (box_list - np.mean(box_list)) / np.std(box_list)

        return box_list, label_list

    def box_dataset_loader(self, file_path):
        # TODO: loader is broken for now, first fix saver, then loader.
        # This method loads a dataset saved in csv format
        read_list = np.genfromtxt(file_path, delimiter=',')
        box_list = []

        for i in range(len(read_list)):  # here we unflatten the dataset for appropriate formats for the NN
            local_box = read_list[i]
            split_location = -self.label_size  # want to split the labels and input, the labels are saved at the end
            box_list.append(Utils.array_split(local_box, split_location))

        return box_list
