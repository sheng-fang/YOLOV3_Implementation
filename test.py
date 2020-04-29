import tensorflow as tf

from core import network


INPUT_SIZE = (416, 416, 3)
NB_CLASS = 20
NB_ANCHOR = 3


# darknet = yolo_network.build_darknet53(INPUT_SIZE, NB_CLASS)
# darknet.summary()
#
# tf.keras.utils.plot_model(darknet, show_shapes=True)


yolo3 = network.build_yolo_network(INPUT_SIZE, NB_CLASS, NB_ANCHOR)
yolo3.summary()
tf.keras.utils.plot_model(yolo3, show_shapes=True)
