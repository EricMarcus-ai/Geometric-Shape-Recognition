import matplotlib.pyplot as plt


def binary_array_plot(array):
    plt.imshow(array, cmap='binary')
    plt.show()