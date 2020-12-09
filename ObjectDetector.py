import DataCreation
import Utils


def main():

    ShapeCreator = DataCreation.ShapeCreator()
    test_box = ShapeCreator.box_creator(False)
    print(test_box[0])
    Utils.binary_array_plot(test_box[0])
    ShapeCreator.box_dataset_creator(1000, 'box_data.csv', True)
    tdat = ShapeCreator.box_dataset_loader('box_data.csv')
    print('hallo')


if __name__ == '__main__':
    main()
