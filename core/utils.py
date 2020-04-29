"""
utils for data preprocessing, IOU, loss function


Sheng FANG
2020-04-28
"""
import cv2
import numpy as np
import tensorflow as tf


def bbox_iou_tf(boxes1, boxes2):
    """
    Calculate iou of the input boxes1 and boxes2, the boxes is in tensorflow format,
    x_center, y_center, w, h
    Args:
        boxes1:
        boxes2:

    Returns:

    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    cornor_boxes1 = tf.concat([boxes1[..., 0: 2] - boxes1[..., 2: 4] * 0.5,
                               boxes1[..., 0: 2] + boxes1[..., 2: 4] * 0.5], axis=-1)

    cornor_boxes2 = tf.concat([boxes2[..., 0: 2] - boxes2[..., 2: 4] * 0.5,
                               boxes2[..., 0: 2] + boxes2[..., 2: 4] * 0.5], axis=-1)

    inter_pnt_tl = tf.maximum(cornor_boxes1[..., 0:2], cornor_boxes2[..., 0:2])
    inter_pnt_br = tf.minimum(cornor_boxes1[..., 2:4], cornor_boxes2[..., 2:4])

    inter_section = tf.maximum(inter_pnt_br - inter_pnt_tl, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def img_preprocess_fix_ratio(img, target_dim, bbox=None):
    """
    resize image to target size without changing the content ratio
    Args:
        image: array
        target_dim: (h, w)
        bbox: array with shape of (n, 4), coordinates xy of top-left and bottom right

    Returns:

    """
    tar_h, tar_w = target_dim
    h, w = img.shape[0: 2]

    h_scale, w_scale = tar_h / h, tar_w / w

    scale = np.min([h_scale, w_scale])

    nh, nw = int(scale * h), int(scale * w)

    img_partial = cv2.resize(img, (nw, nh))

    img_resize = np.ones([tar_h, tar_w, 3], dtype=np.float32) * 128.0
    dw = int((tar_w - nw) / 2)
    dh = int((tar_h - nh) / 2)
    img_resize[dh: nh + dh, dw: nw + dw, :] = img_partial

    if bbox is not None:
        bbox = bbox * scale + np.array([dw, dh, dw, dh]).reshape((1, 4))

    return img_resize, bbox









