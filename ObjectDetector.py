from ShapeCreator import ShapeCreator
import Utils
from Defines import background_size, num_objects
from FeedForwardModel import FFM
from ConvolutionalModel import CNN


def main():

    # Create the datasets, flatten=False for Convolutional and True for FeedForward
    train_data_creator = ShapeCreator(background_size=background_size)
    train_x, train_y = train_data_creator.dataset_creator(10000, flatten=False)
    val_x, val_y = train_data_creator.dataset_creator(5000, flatten=False)
    test_x, test_y = train_data_creator.dataset_creator(2000, flatten=False)
    input_size = train_x.shape[1:]

    # Training sets can also be loaded from saved files
    # train_x, train_y = train_data_creator.dataset_loader('box_train_data')
    # val_x, val_y = train_data_creator.dataset_loader('box_val_data')
    # test_x, test_y = train_data_creator.dataset_loader('box_test_data')

    # The feedforward (or fully connected) model
    # n = FFM(input_size=input_size)
    # n.initialize_model()
    # n.train_model(train_x, train_y, val_x, val_y, batch_size=1024, epochs=100)
    # n.test_model(test_x, test_y)

    # The convolutional model
    n = CNN(input_size=input_size)
    n.initialize_model()
    n.train_model(train_x, train_y, val_x, val_y, batch_size=512, epochs=100)
    n.test_model(test_x, test_y)

    # preds = n.prediction(test_x[0:10])
    # Utils.draw_predictions(test_x[0:10], preds, true_labels=test_y[0:10])

    # Create data with multiple objects per image
    multi_object_creator = ShapeCreator(background_size=background_size, num_objects=num_objects)
    multi_dat, multi_label = multi_object_creator.dataset_creator(10)

    # Predict on the multi-object dataset
    multi_preds = n.multi_object_prediction(multi_dat)
    Utils.draw_predictions(multi_dat, multi_preds)

    print('hallo')


if __name__ == '__main__':
    main()
