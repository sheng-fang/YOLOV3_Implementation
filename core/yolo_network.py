"""
Implementation of Darknet-53 and YOLOV3 network
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, UpSampling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import Input, Dense, ZeroPadding2D, LeakyReLU
from tensorflow.keras.models import Model


def conv_block(x, cfg):
    conv = x



def conv_bn_relu(x, filters, kernel_size, strides=1, padding="same", alpha=0.1):
    """
    Basic unit to do convolution, batch normalization, and leaky relu
    Args:
        x:
        filters:
        kernel_size:
        strides:
        padding:
        alpha:

    Returns:

    """
    if strides > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = Conv2D(filters, kernel_size, strides=strides, padding="valid")(x)
    else:
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x, alpha)

    return x


def residual_block(x, filters, kernel_size, strides=1, padding="same", alpha=0.1):
    """
    Residual block defined in DarkNet-53
    Args:
        x:
        filters:
        kernel_size:
        strides:
        padding:
        alpha:

    Returns:

    """
    res = conv_bn_relu(x, int(filters/2), 1, strides, padding, alpha)
    res = conv_bn_relu(res, filters, kernel_size, strides, padding, alpha)
    x = Add()([res, x])

    return x


def res_repeat(nb_res_block, x, filters, kernel_size, strides=1, padding="same", alpha=0.1):
    """
    Repeat darknet residual block several times.
    Args:
        nb_res_block:
        x:
        filters:
        kernel_size:
        strides:
        padding:
        alpha:

    Returns:

    """
    for idx in range(nb_res_block):
        x = residual_block(x, filters, kernel_size, strides, padding, alpha)

    return x


def build_darknet53_backbone(input_size):
    """
    Build DarkNet-53 body
    Args:
        input_size:
        nb_class:

    Returns:

    """
    x = inputs = Input(shape=input_size)
    x = conv_bn_relu(x, filters=32, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, filters=64, kernel_size=3, strides=2, padding="same", alpha=0.1)
    x = res_repeat(1, x, filters=64, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, filters=128, kernel_size=3, strides=2, padding="same", alpha=0.1)
    x = res_repeat(2, x, filters=128, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, filters=256, kernel_size=3, strides=2, padding="same", alpha=0.1)
    stride8 = res_repeat(8, x, filters=256, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(stride8, filters=512, kernel_size=3, strides=2, padding="same", alpha=0.1)
    stride16 = res_repeat(8, x, filters=512, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(stride16, filters=1024, kernel_size=3, strides=2, padding="same", alpha=0.1)
    stride32 = res_repeat(4, x, filters=1024, kernel_size=3, strides=1, padding="same", alpha=0.1)
    # tf.print(stride8.shape)
    # tf.print(stride16.shape)
    # tf.print(stride32.shape)

    return inputs, stride32, stride16, stride8


def build_darknet53_top(x, nb_class):
    """
    Build DarkNet-53 top
    Args:
        x:
        nb_class:

    Returns:

    """
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=1000)(x)
    x = tf.keras.activations.relu(x, alpha=0.1)
    x = Dense(units=nb_class, activation="softmax")(x)

    return x


def build_darknet53(input_size, nb_calss):
    """
    Build darknet53
    Args:
        input_size:
        nb_calss:

    Returns:

    """
    inputs, stride32, stride16, stride8 = build_darknet53_backbone(input_size)
    outputs = build_darknet53_top(stride32, nb_calss)

    darknet = Model(inputs=inputs, outputs=outputs)

    return darknet


def build_yolo_network(input_size, nb_class, nb_archor):
    cell_pred_dim = nb_archor * (5 + nb_class)
    inputs, stride32, stride16, stride8 = build_darknet53_backbone(input_size)

    x = conv_bn_relu(stride32, 512, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 1024, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 512, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 1024, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 512, kernel_size=1, strides=1, padding="same", alpha=0.1)

    lobj_pred = conv_bn_relu(x, 1024, kernel_size=3, strides=1, padding="same", alpha=0.1)
    lobj_pred = Conv2D(filters=cell_pred_dim, kernel_size=1, strides=1, padding="same")(lobj_pred)

    x = conv_bn_relu(x, 256, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = UpSampling2D(2)(x)
    x = tf.concat([stride16, x], axis=-1)

    x = conv_bn_relu(x, 256, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 512, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 256, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 512, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 256, kernel_size=1, strides=1, padding="same", alpha=0.1)

    mobj_pred = conv_bn_relu(x, 512, kernel_size=3, strides=1, padding="same", alpha=0.1)
    mobj_pred = Conv2D(filters=cell_pred_dim, kernel_size=1, strides=1, padding="same")(mobj_pred)

    x = conv_bn_relu(x, 128, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = UpSampling2D(2)(x)
    x = tf.concat([stride8, x], axis=-1)

    x = conv_bn_relu(x, 128, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 256, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 128, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 256, kernel_size=3, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 128, kernel_size=1, strides=1, padding="same", alpha=0.1)
    x = conv_bn_relu(x, 256, kernel_size=3, strides=1, padding="same", alpha=0.1)
    sobj_pred = Conv2D(filters=cell_pred_dim, kernel_size=1, strides=1, padding="same")(x)

    yolo = Model(inputs=inputs, outputs=[lobj_pred, mobj_pred, sobj_pred])

    return yolo
