import DataCreation
import Utils
from FeedForwardModel import FFM


def main():
    """This is where the magic happens"""
    #  TODO: implement custom keras loss function by IOU
    #   Also perhaps check if there are different results with pytorch or sklearn
    #   in terms of the learning performance

    shape_creator = DataCreation.ShapeCreator()
    test_box = shape_creator.box_creator(flatten=False)

    # Utils.binary_array_plot(test_box[0])
    train_x, train_y = shape_creator.box_dataset_creator(100000, 'box_data.csv')
    test_x, test_y = shape_creator.box_dataset_creator(10000, 'test_box_set.csv')
    # tdat = shape_creator.box_dataset_loader('box_data.csv')

    n = FFM()
    n.initialize_model()
    n.train_model(train_x, train_y, 'objectdetect', batch_size=1000, epochs=10)
    n.test_model(test_x, test_y)

    # print(test_box[0])
    # pred = n.prediction(test_box[0].flatten())
    # print(pred, test_box[1])
    # Utils.binary_array_plot(test_box[0])
    # Utils.draw_rectangle(test_box[1])
    # Utils.draw_rectangle(pred, colour='b')
    preds = n.prediction(test_x[0:10])
    # TODO: fix the draw_predictions or change input to non flattened
    Utils.draw_predictions(test_x[0:10], test_y[0:10], preds)
    print('hallo')


if __name__ == '__main__':
    main()
