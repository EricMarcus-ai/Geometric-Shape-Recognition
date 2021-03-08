from ShapeCreator import ShapeCreator
import Utils
from Defines import background_size, num_objects, label_size
from FeedForwardModel import FFM


def main():
    """This is where the magic happens"""
    #  TODO: Create openCV / skimage features to extract seperately the boxes in an image and feed after one another
    #   This project was built as a precursor to plaque detector for alzheimer
    #   Include another type of object, e.g. circle

    # TODO: Train first solo object classifier, then apply the network to multi objects after

    train_data_creator = ShapeCreator(background_size=background_size)
    train_x, train_y = train_data_creator.box_dataset_creator(20000)
    val_x, val_y = train_data_creator.box_dataset_creator(2000)

    # train_x, train_y = train_data_creator.box_dataset_loader('box_train_data')
    # val_x, val_y = train_data_creator.box_dataset_loader('box_val_data')

    test_data_creator = ShapeCreator(background_size=background_size, num_objects=num_objects, label_size=label_size)
    test_x, test_y = train_data_creator.box_dataset_creator(2000)

    # test_x, test_y = train_data_creator.box_dataset_loader('box_test_data')

    n = FFM(background_size=background_size, num_objects=num_objects, label_size=label_size)
    n.initialize_model()
    n.train_model(train_x, train_y, val_x, val_y, batch_size=1024, epochs=100)
    n.test_model(test_x, test_y)

    preds = n.prediction(test_x[0:10])
    preds2 = n.prediction(train_x[0:10])
    Utils.draw_predictions(test_x[0:10], test_y[0:10], preds)
    Utils.draw_predictions(train_x[0:10], train_y[0:10], preds2)
    print('hallo')


if __name__ == '__main__':
    main()
