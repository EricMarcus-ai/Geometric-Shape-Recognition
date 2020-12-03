import numpy as np
from Defines import background_size
import random


class ShapeCreator:

    def __init__(self):
        self.back_size = background_size

    def box_creator(self):
        box_array = np.zeros((self.back_size, self.back_size))  # creating background
        rand_ints = sorted([random.randint(0, self.back_size) for i in range(4)])  # generate random points for box

        while rand_ints[-2] == rand_ints[0]:  # we need two different points for a non-empty box (also no line => [-2])
            rand_ints = sorted([random.randint(0, self.back_size) for i in range(4)])

        sampled_points = rand_ints  # below we make sure that there are no two consecutive equal numbers

        while sampled_points[0] == sampled_points[1] or sampled_points[2] == sampled_points[3]:
            sampled_points = random.sample(rand_ints, 4)

        bound_list1 = sorted(sampled_points[0:2])  # these bounding lists now consist out of two different points
        bound_list2 = sorted(sampled_points[2:4])

        box_array[bound_list1[0]:bound_list1[1], bound_list2[0]:bound_list2[1]] = 1
        x_coord = sampled_points[0]
        y_coord = sampled_points[2]
        width = sampled_points[1] - x_coord
        height = sampled_points[3] - y_coord

        return [box_array, [x_coord, y_coord, width, height]]

    def box_dataset_creator(self, num_imgs):
        # TODO: loop over box creator and make like 40-50k set and learn how to save this