from Defines import background_size, num_boxes, label_size
import matplotlib.pyplot as plt
import matplotlib.patches as patches


"""The Utils file contains various utility functions"""


# This function creates a plot of a np.array
def binary_array_plot(array):
    plt.imshow(array, cmap='Greys', interpolation='none', origin='lower', extent=[0, background_size, 0, background_size])
    plt.show(block=False)


# This function draws a rectangle on inputted coordinates
def draw_rectangle(coords, colour='r'):
    coords = background_size * coords.flatten()
    rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=1, edgecolor=colour, facecolor='none')
    plt.gca().add_patch(rect)
    plt.show(block=False)


# This function plots the prediction and true labels through inputted data
def draw_predictions(input_boxes, input_labels, predicted_labels):

    inp_len = len(input_boxes)
    if input_boxes.shape == (inp_len, background_size * background_size):  # reshape into plottable shape
        input_boxes = input_boxes.reshape(inp_len, background_size, background_size)

    for i in range(len(input_boxes)):  # loop over all the different examples
        plt.clf()  # clear previous plot
        binary_array_plot(input_boxes[i])  # plot background + box(es)

        for j in range(num_boxes):  # loop over the different boxes in each example
            index_start = j * label_size  # these two indices pick out the individual box coordinates
            index_end = j * label_size + label_size

            true_rect = input_labels[i][index_start:index_end]  # these two rectangles are the true and predicted labels
            pred_rect = predicted_labels[i][index_start:index_end]

            draw_rectangle(true_rect)  # plot true rectangle
            draw_rectangle(pred_rect, colour='b')  # plot predicted rectangle

        plt.show()
        inp = input("Press enter for continue or q for quit")
        if inp == 'q':
            return

    return


# This function splits the array at split_loc
def array_split(input_array, split_loc):
    split_array = input_array[0:split_loc], input_array[split_loc:]
    return split_array
