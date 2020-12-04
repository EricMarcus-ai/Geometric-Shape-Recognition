import DataCreation
import Utils

ShapeCreator = DataCreation.ShapeCreator()
test_box = ShapeCreator.box_creator()
#print(test_box)
#Utils.binary_array_plot(test_box)
ShapeCreator.box_dataset_creator(1000, 'box_data.csv', True)
tdat = ShapeCreator.box_dataset_loader('box_data.csv')
print('hallo')
