"""
Implementation of Darknet-53 and YOLOV3 network
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tf_layers
import tensorflow.keras.models as tf_models

from core import utils


ANCHOR_CFG = "config/anchor_cfg.txt"
STRIDE_CFG = "config/strides.txt"

ANCHORS = tf.convert_to_tensor(utils.load_anchors(ANCHOR_CFG), dtype=tf.int32)
STRIDES = tf.convert_to_tensor(np.loadtxt(STRIDE_CFG, delimiter=","),
                               dtype=tf.int32)


# pylint: disable=too-many-arguments
def conv_bn_relu(input_tensor, filters, kernel_size, strides=1, padding="same",
                 alpha=0.1):
    """Basic unit to do convolution, batch normalization, and leaky relu

    Args:
        input_tensor:
        filters:
        kernel_size:
        strides:
        padding:
        alpha:

    Returns:

    """
    if strides > 1:
        input_tensor = tf_layers.ZeroPadding2D(((1, 0), (1, 0)))(input_tensor)
        input_tensor = tf_layers.Conv2D(filters, kernel_size, strides=strides,
                                        padding="valid",)(input_tensor)
    else:
        input_tensor = tf_layers.Conv2D(filters, kernel_size, strides=strides,
                                        padding=padding)(input_tensor)
    input_tensor = tf_layers.BatchNormalization()(input_tensor)
    input_tensor = tf_layers.LeakyReLU(alpha)(input_tensor)

    return input_tensor


def residual_block(input_tensor, filters, kernel_size, strides=1,
                   padding="same", alpha=0.1):
    """
    Residual block defined in DarkNet-53
    Args:
        input_tensor:
        filters:
        kernel_size:
        strides:
        padding:
        alpha:

    Returns:

    """
    res = conv_bn_relu(input_tensor, int(filters/2), 1, strides, padding, alpha)
    res = conv_bn_relu(res, filters, kernel_size, strides, padding, alpha)
    input_tensor = tf_layers.Add()([res, input_tensor])

    return input_tensor


def res_repeat(nb_res_block, input_tensor, filters, kernel_size, strides=1,
               padding="same", alpha=0.1):
    """
    Repeat darknet residual block several times.
    Args:
        nb_res_block:
        input_tensor:
        filters:
        kernel_size:
        strides:
        padding:
        alpha:

    Returns:

    """
    for _ in range(nb_res_block):
        input_tensor = residual_block(input_tensor, filters, kernel_size,
                                      strides, padding, alpha)

    return input_tensor


def build_darknet53_backbone(input_size):
    """
    Build DarkNet-53 body
    Args:
        input_size:

    Returns:

    """
    tmp_tensor = inputs = tf_layers.Input(shape=input_size)
    tmp_tensor = conv_bn_relu(tmp_tensor, filters=32, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, filters=64, kernel_size=3, strides=2,
                              padding="same", alpha=0.1)
    tmp_tensor = res_repeat(1, tmp_tensor, filters=64, kernel_size=3, strides=1,
                            padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, filters=128, kernel_size=3, strides=2,
                              padding="same", alpha=0.1)
    tmp_tensor = res_repeat(2, tmp_tensor, filters=128, kernel_size=3,
                            strides=1, padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, filters=256, kernel_size=3, strides=2,
                              padding="same", alpha=0.1)
    stride8 = res_repeat(8, tmp_tensor, filters=256, kernel_size=3, strides=1,
                         padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(stride8, filters=512, kernel_size=3, strides=2,
                              padding="same", alpha=0.1)
    stride16 = res_repeat(8, tmp_tensor, filters=512, kernel_size=3, strides=1,
                          padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(stride16, filters=1024, kernel_size=3, strides=2,
                              padding="same", alpha=0.1)
    stride32 = res_repeat(4, tmp_tensor, filters=1024, kernel_size=3, strides=1,
                          padding="same", alpha=0.1)

    return inputs, stride32, stride16, stride8


def build_darknet53_top(tmp_tensor, nb_class):
    """
    Build DarkNet-53 top
    Args:
        tmp_tensor:
        nb_class:

    Returns:

    """
    tmp_tensor = tf_layers.GlobalAveragePooling2D()(tmp_tensor)
    tmp_tensor = tf_layers.Flatten()(tmp_tensor)
    tmp_tensor = tf_layers.Dense(units=1000)(tmp_tensor)
    tmp_tensor = tf_layers.LeakyReLU(alpha=0.1)(tmp_tensor)
    tmp_tensor = tf_layers.Dense(units=nb_class,
                                 activation="softmatmp_tensor")(tmp_tensor)

    return tmp_tensor


def build_darknet53(input_size, nb_calss):
    """
    Build darknet53
    Args:
        input_size:
        nb_calss:

    Returns:

    """
    inputs, stride32, _, _ = build_darknet53_backbone(input_size)
    outputs = build_darknet53_top(stride32, nb_calss)

    darknet = tf_models.Model(inputs=inputs, outputs=outputs)

    return darknet


def build_yolo_network(input_size, nb_class, anchor_per_cell):
    """

    Args:
        input_size:
        nb_class:
        anchor_per_cell:

    Returns:

    """
    cell_pred_dim = anchor_per_cell * (5 + nb_class)
    inputs, stride32, stride16, stride8 = build_darknet53_backbone(input_size)

    tmp_tensor = conv_bn_relu(stride32, 512, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 1024, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 512, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 1024, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 512, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)

    lobj_pred = conv_bn_relu(tmp_tensor, 1024, kernel_size=3, strides=1,
                             padding="same", alpha=0.1)
    lobj_pred = tf_layers.Conv2D(filters=cell_pred_dim, kernel_size=1,
                                 strides=1, padding="same")(lobj_pred)

    tmp_tensor = conv_bn_relu(tmp_tensor, 256, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = tf_layers.UpSampling2D(2)(tmp_tensor)
    tmp_tensor = tf.concat([stride16, tmp_tensor], atmp_tensoris=-1)

    tmp_tensor = conv_bn_relu(tmp_tensor, 256, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 512, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 256, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 512, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 256, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)

    mobj_pred = conv_bn_relu(tmp_tensor, 512, kernel_size=3, strides=1,
                             padding="same", alpha=0.1)
    mobj_pred = tf_layers.Conv2D(filters=cell_pred_dim, kernel_size=1,
                                 strides=1, padding="same")(mobj_pred)

    tmp_tensor = conv_bn_relu(tmp_tensor, 128, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = tf_layers.UpSampling2D(2)(tmp_tensor)
    tmp_tensor = tf.concat([stride8, tmp_tensor], atmp_tensoris=-1)

    tmp_tensor = conv_bn_relu(tmp_tensor, 128, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 256, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 128, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 256, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 128, kernel_size=1, strides=1,
                              padding="same", alpha=0.1)
    tmp_tensor = conv_bn_relu(tmp_tensor, 256, kernel_size=3, strides=1,
                              padding="same", alpha=0.1)
    sobj_pred = tf_layers.Conv2D(filters=cell_pred_dim, kernel_size=1,
                                 strides=1, padding="same")(tmp_tensor)

    lobj_pred = yolo_decode(lobj_pred, nb_class, anchor_per_cell)
    mobj_pred = yolo_decode(mobj_pred, nb_class, anchor_per_cell)
    sobj_pred = yolo_decode(sobj_pred, nb_class, anchor_per_cell)

    yolo = tf_models.Model(inputs=inputs,
                           outputs=[lobj_pred, mobj_pred, sobj_pred])

    return yolo


def yolo_decode(conv_output, nb_class, anchor_per_cell):
    """
    Decode the convolutional output by adding sigmoid and power activation
    Args:
        conv_output:
        nb_class:
        anchor_per_cell:

    Returns:

    """
    batch_size, output_size = tf.shape(conv_output)[0:2]

    curr_output = tf.reshape(conv_output, (batch_size, output_size, output_size,
                                           anchor_per_cell, 5 + nb_class)
                             )

    dxdy = curr_output[..., 0:2]
    wh_percent = curr_output[..., 2:4]
    conf_prob = curr_output[..., 4:]

    dxdy = tf.sigmoid(dxdy)
    wh_percent = tf.exp(wh_percent)
    conf_prob = tf.sigmoid(conf_prob)

    return tf.concat([dxdy, wh_percent, conf_prob], axis=-1)


def compute_loss(y_true, y_pred):
    """

    Args:
        y_true:
        y_pred:

    Returns:

    """
    batch_size, output_size = tf.shape(y_pred)[0:2]

    xywh_pred = y_pred[..., 0:4]
    conf_pred = y_pred[..., 4:5]
    prob_pred = y_pred[..., 5:]

    xywh_gt = y_true[..., 0:4]
    response_gt = y_true[..., 4:5]
    prob_gt = y_true[..., 5:]

    loss_scale = 2.0 - xywh_gt[..., 2:3] * xywh_gt[..., 3:4]
    # box loss
    xywh_loss = response_gt * loss_scale * tf.square(xywh_pred - xywh_gt)
    # objectness loss
    obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(response_gt, conf_pred)
    # class loss
    cls_loss = response_gt * tf.square(prob_gt - prob_pred)

    loss = (tf.reduce_sum(xywh_loss) + tf.reduce_sum(obj_loss)
            + tf.reduce_sum(cls_loss))

    return loss / tf.cast(batch_size * output_size * output_size, tf.float32)
