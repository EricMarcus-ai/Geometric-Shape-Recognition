from Defines import background_size, num_objects
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
    label_size = input_labels.shape[-1]  # lenght of a single label, used in loops below
    inp_len = len(input_boxes)

    if input_boxes.shape == (inp_len, background_size * background_size):  # unflatten
        input_boxes = input_boxes.reshape(inp_len, background_size, background_size)

    for i in range(len(input_boxes)):
        plt.clf()  # clear previous plot
        binary_array_plot(input_boxes[i])

        for j in range(num_objects):
            index_start = j * label_size  # these two indices pick out the individual box coordinates
            index_end = j * label_size + label_size

            true_rect = input_labels[i][index_start:index_end]  # these two rectangles are the true and predicted labels
            pred_rect = predicted_labels[i][index_start:index_end]

            draw_rectangle(true_rect)  # plot true rectangle
            draw_rectangle(pred_rect, colour='b')  # plot predicted rectangle

        plt.show()
        # inp = input("Press enter for continue or q for quit")
        # if inp == 'q':
        #     break


# This function splits the array at split_loc
def array_split(input_array, split_loc):
    split_array = input_array[0:split_loc], input_array[split_loc:]
    return split_array
