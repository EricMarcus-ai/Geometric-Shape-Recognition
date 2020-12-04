import numpy as np
from Defines import background_size
import random
import pandas as pd
import csv


class ShapeCreator:

    def __init__(self):
        self.back_size = background_size

    def box_creator(self, flatten=False):
        # TODO: implement the x,y,w,h earlier on, now I define them at the end
        box_array = np.zeros((self.back_size, self.back_size))  # creating background

        rand1 = sorted([random.randint(0, self.back_size) for i in range(2)])

        if rand1[0] == rand1[1]:
            allowed_numbers = np.setdiff1d(np.arange(0, self.back_size), rand1[0]).tolist()
            rand2 = random.choices(allowed_numbers, k=2)

        else:
            allowed_numbers = np.arange(0, self.back_size).tolist()
            rand2 = sorted(random.sample(allowed_numbers, 2))

        sampled_points = rand1 + rand2  # below we make sure that there are no two consecutive equal numbers

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

        box_list = []
        for i in range(num_imgs):
            local_box = self.box_creator(True)
            box_list.append(np.hstack(local_box))  # hstack flattens the list, when loading we have to repack

        if saved:
            np.savetxt(filename, box_list, delimiter=',')

        return box_list

    def box_dataset_loader(self, file_path):
        box_list = np.genfromtxt(file_path, delimiter=',')

        return box_list
