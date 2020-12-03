import DataCreation
import Utils

ShapeCreator = DataCreation.ShapeCreator()
test_box = ShapeCreator.box_creator()
Utils.binary_array_plot(test_box)
