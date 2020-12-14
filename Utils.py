from Defines import background_size
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    for i in range(len(input_boxes)):
        binary_array_plot(input_boxes[i])
        draw_rectangle(input_labels[i])
        draw_rectangle(predicted_labels[i], colour='b')
        inp = input("Press enter for continue or q for quit")
        if inp == 'q':
            return

    return


# This function splits the array at split_loc
def array_split(input_array, split_loc):
    split_array = input_array[0:split_loc], input_array[split_loc:]
    return split_array
