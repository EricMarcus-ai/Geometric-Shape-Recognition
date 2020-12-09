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
        # this method creates one box object
        # TODO: implement the x,y,w,h earlier on, now I define them at the end
        box_array = np.zeros((self.back_size, self.back_size))  # creating background

        rand1 = sorted([random.randint(0, self.back_size) for i in range(2)])  # first random pair

        if rand1[0] == rand1[1]:   # check if the random numbers are the same, which is important for the next numbers
            allowed_numbers = np.setdiff1d(np.arange(0, self.back_size), rand1[0]).tolist()
            rand2 = random.choices(allowed_numbers, k=2)

        else:
            allowed_numbers = np.arange(0, self.back_size).tolist()
            rand2 = sorted(random.sample(allowed_numbers, 2))

        sampled_points = rand1 + rand2  # collection of all random sampled allowed pairs (no 3 points equal)

        #  below we make sure that the first-second or third-fourth pairs are unequal, because they will describe
        #  the bounding boxes. Equal numbers would result in a box with a zero width or height.
        while sampled_points[0] == sampled_points[1] or sampled_points[2] == sampled_points[3]:
            sampled_points = random.sample(sampled_points, 4)

        bound_list1 = sorted(sampled_points[0:2])  # these bounding lists now consist out of two different points
        bound_list2 = sorted(sampled_points[2:4])

        box_array[bound_list1[0]:bound_list1[1], bound_list2[0]:bound_list2[1]] = 1
        x_coord = bound_list1[0]
        y_coord = bound_list2[0]
        height = bound_list1[1] - x_coord
        width = bound_list2[1] - y_coord

        if flatten:
            return box_array.flatten(), np.array([x_coord, y_coord, width, height])
        else:
            return [box_array, np.array([[x_coord, y_coord, width, height]])]

    def box_dataset_creator(self, num_imgs, filename, saved=False):
        # This method always returns a non-flattened box_list and (if passed) saves a flattened version in csv
        box_list = []
        for i in range(num_imgs):
            local_box = self.box_creator()
            box_list.append(local_box)  # hstack flattens the list, when loading we have to repack

        if saved:  # if saved, we save the box_list flattened in csv format
            save_list = []
            for i in range(len(box_list)):
                save_list.append(np.hstack(box_list[i]))
            np.savetxt(filename, save_list, delimiter=',')

        return box_list

    def box_dataset_loader(self, file_path):
        #  This method loads a dataset saved in csv format
        read_list = np.genfromtxt(file_path, delimiter=',')
        box_list = []

        for i in range(len(read_list)):  # here we unflatten the dataset for appropriate formats for the NN
            local_box = read_list[i]
            split_location = -self.label_size  # want to split the labels and input, the labels are saved at the end
            box_list.append(Utils.array_split(local_box, split_location))

        return box_list
