from Defines import background_size
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Defines import num_possible_shapes


# Creates a simple plot of a np.array
def binary_array_plot(array):
    if array.shape == (background_size, background_size, 1):  # remove channel
        array = array.reshape(background_size, background_size)
    plt.imshow(array, cmap='Greys', interpolation='none', origin='lower', extent=[0, background_size, 0, background_size])
    plt.show(block=False)


# Draws a rectangle on inputted coordinates
def draw_rectangle(coords, color='r', object_type=''):
    coords = background_size * coords.flatten()
    rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=1, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)
    if object_type:
        text_x = coords[0] + coords[2] / 3
        text_y = coords[1] + coords[3]
        plt.text(text_x, text_y, object_type, color=color)
    plt.show(block=False)


# Unpacks and processes the label into object types and bounding boxes
def unpack_label(labels):
    label_length = num_possible_shapes + 4  # bbox size is 4
    if labels.shape == (label_length,):  # for single inputs we don't need to flatten input
        pass
    else:
        labels = np.concatenate(labels)
    object_dict = {0: 'box', 1: 'circle', 2: 'triangle'}
    num_labels = int(len(labels) / label_length)

    object_list = []
    bbox_list = []
    current_label = 0
    for i in range(num_labels * 2):  # iterate to 2 * num_labels to alternate between object and bbox predictions
        if i % 2 == 0:
            local_object = labels[current_label:current_label + num_possible_shapes]  # extract the object prediction
            obj_pred = np.argmax(local_object).item()  # cast as int using item
            obj_name = object_dict[obj_pred]
            object_list.append(obj_name)
            current_label += 3
        else:
            local_bbox = labels[current_label:current_label + 4]
            bbox_list.append(local_bbox)
            current_label += 4
    return object_list, np.array(bbox_list)


# Plots the prediction and possibly the true labels on top of input image
def draw_predictions(input_images, predicted_labels, true_labels=np.array([])):
    if input_images.shape == (len(input_images), background_size ** 2):  # unflatten if necessary
        input_images = input_images.reshape(len(input_images), background_size, background_size)

    object_types_p, bboxes_p = unpack_label(predicted_labels)
    if true_labels.any():
        object_types_t, bboxes_t = unpack_label(true_labels)

    bbox_size = 4
    label_size = bbox_size + num_possible_shapes
    index = 0  # used for keeping track of bboxes
    for i in range(len(input_images)):
        plt.clf()  # clear previous plot
        binary_array_plot(input_images[i])
        number_objects = int(len(predicted_labels[i]) / label_size)  # how many objects did the NN predict

        for j in range(number_objects):
            pred_rect = bboxes_p[index]
            draw_rectangle(pred_rect, color='b', object_type=object_types_p[index])
            if true_labels.any():
                true_rect = bboxes_t[index]
                draw_rectangle(true_rect, object_type=object_types_t[index])
            index += 1
        plt.show()


# Calculate Intersection Over Union
def intersection_over_union(true_labels, predicted_labels, print_result=True):
    _, bbox_true = unpack_label(true_labels)
    _, bbox_pred = unpack_label(predicted_labels)
    num_bbox = len(bbox_true)
    if len(bbox_pred) != len(bbox_true):
        raise ValueError('Unequal amount of bboxes found between predicted_labels and true_labels')

    iou_list = []
    for i in range(num_bbox):
        bbox_true[i][2] += bbox_true[i][0]  # convert width to x2 coordinate
        bbox_true[i][3] += bbox_true[i][1]  # convert height to y2 coordinate
        bbox_pred[i][2] += bbox_pred[i][0]
        bbox_pred[i][3] += bbox_pred[i][1]

        (x1_t, y1_t, x2_t, y2_t) = bbox_true[i]
        (x1_p, y1_p, x2_p, y2_p) = bbox_pred[i]
        xi1 = max(x1_t, x1_p)
        yi1 = max(y1_t, y1_p)
        xi2 = min(x2_t, x2_p)
        yi2 = min(y2_t, y2_p)

        inter_area = max(xi2-xi1, 0) * max(yi2 - yi1, 0)
        true_area = (x2_t - x1_t) * (y2_t - y1_t)
        pred_area = (x2_p - x1_p) * (y2_p - y1_p)
        iou = inter_area / (true_area + pred_area - inter_area)
        iou_list.append(iou)

    mean_iou = np.mean(iou_list)
    if print_result:
        print('IOU over ' + str(num_bbox) + ' bboxes is found to be ' + str(mean_iou))
    return mean_iou
