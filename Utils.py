from Defines import background_size
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

    for i in range(len(input_boxes)):
        plt.clf()  # clear previous plot
        binary_array_plot(input_boxes[i])  # plot box
        draw_rectangle(input_labels[i])  # plot true box
        draw_rectangle(predicted_labels[i], colour='b')  # plot predicted box
        plt.show()
        inp = input("Press enter for continue or q for quit")
        if inp == 'q':
            return

    return


# This function splits the array at split_loc
def array_split(input_array, split_loc):
    split_array = input_array[0:split_loc], input_array[split_loc:]
    return split_array
