import DataCreation
import Utils
from FeedForwardModel import FFM


def main():
    """This is where the magic happens"""
    #  TODO: think about the normalization and mean averaging for training versus test data
    #   Currently I do them all individually but this is  not right.
    #   Should I init in the datacreation with the self.blabla, when the things are already imported from defines

    shape_creator = DataCreation.ShapeCreator()

    train_x, train_y = shape_creator.box_dataset_creator(60000)
    val_x, val_y = shape_creator.box_dataset_creator(20000)
    test_x, test_y = shape_creator.box_dataset_creator(20000)
    # train_x, train_y = shape_creator.box_dataset_loader('box_train_data')
    # val_x, val_y = shape_creator.box_dataset_loader('box_val_data')
    # test_x, test_y = shape_creator.box_dataset_loader('box_test_data')

    n = FFM()
    n.initialize_model()
    n.train_model(train_x, train_y, val_x, val_y, batch_size=1000, epochs=100)
    n.test_model(test_x, test_y)

    preds = n.prediction(test_x[0:10])
    preds2 = n.prediction(train_x[0:10])
    Utils.draw_predictions(test_x[0:10], test_y[0:10], preds)
    Utils.draw_predictions(train_x[0:10], train_y[0:10], preds2)
    print('hallo')


if __name__ == '__main__':
    main()
