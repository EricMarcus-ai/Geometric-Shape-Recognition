import matplotlib.pyplot as plt


# This function creates a plot of a np.array
def binary_array_plot(array):
    plt.imshow(array, cmap='binary')
    plt.show()


# This function splits the array at split_loc
def array_split(input_array, split_loc):
    split_array = input_array[0:split_loc], input_array[split_loc:]
    return split_array
