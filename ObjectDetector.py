import DataCreation
import Utils
from FeedForwardModel import FFM


def main():
    """This is where the magic happens"""
    #  TODO: Create openCV / skimage features to extract seperately the boxes in an image and feed after one another
    #   Make background larger to diminish overlapping chance
    #   This project was built as a precursor to plaque detector for alzheimer
    #   Include another type of object, e.g. circle
    #   Should I init in the datacreation with the self.blabla, when the things are already imported from defines

    # TODO: Train first solo object classifier, then apply the network to multi objects after
    shape_creator = DataCreation.ShapeCreator()

    train_x, train_y = shape_creator.box_dataset_creator(60000)
    val_x, val_y = shape_creator.box_dataset_creator(20000)
    test_x, test_y = shape_creator.box_dataset_creator(20000)
    # train_x, train_y = shape_creator.box_dataset_loader('box_train_data')
    # val_x, val_y = shape_creator.box_dataset_loader('box_val_data')
    # test_x, test_y = shape_creator.box_dataset_loader('box_test_data')

    n = FFM()
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
