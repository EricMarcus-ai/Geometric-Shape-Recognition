from ShapeCreator import ShapeCreator
import Utils
from Defines import background_size, num_objects
from FeedForwardModel import FFM
from ConvolutionalModel import CNN


def main():
    """ Contains data creation, neural network training and multi object prediction """

    # Create the datasets, flatten=False for Convolutional and True for FeedForward
    data = ShapeCreator(background_size=background_size)
    train_x, train_y = data.dataset_creator(10000, flatten=False)
    val_x, val_y = data.dataset_creator(5000, flatten=False)
    test_x, test_y = data.dataset_creator(2000, flatten=False)
    input_size = train_x.shape[1:]

    # Training sets can also be loaded from saved files
    # train_x, train_y = data.dataset_loader('box_train_data')
    # val_x, val_y = data.dataset_loader('box_val_data')
    # test_x, test_y = data.dataset_loader('box_test_data')

    # The feedforward (or fully connected) model
    # n = FFM(input_size=input_size)
    # n.initialize_model()
    # n.train_model(train_x, train_y, val_x, val_y, batch_size=512, epochs=100)
    # n.test_model(test_x, test_y)

    # The convolutional model
    n = CNN(input_size=input_size)
    n.initialize_model()
    n.train_model(train_x, train_y, val_x, val_y, batch_size=512, epochs=150, model_name='CNN_Model.h5')
    # n.save_model('CNN_Model.h5')
    n.test_model(test_x, test_y)

    # Single object predictions
    preds = n.prediction(test_x[0:100])
    Utils.intersection_over_union(test_y[0:100], preds)
    # Utils.draw_predictions(test_x[0:10], preds, true_labels=test_y[0:10])

    # Create data with multiple objects per image
    multi_object = ShapeCreator(background_size=background_size, num_objects=num_objects)
    multi_dat, multi_label = multi_object.dataset_creator(10, flatten=False)

    # Predict on the multi-object dataset (flatten = False for Convolutional)
    multi_preds = n.multi_object_prediction(multi_dat, flatten=False)
    Utils.draw_predictions(multi_dat, multi_preds)


if __name__ == '__main__':
    main()
