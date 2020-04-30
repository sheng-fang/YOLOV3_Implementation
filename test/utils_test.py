"""Unittest

"""
import unittest

import numpy as np
import tensorflow as tf

from core import utils


class MyTestCase(unittest.TestCase):
    """
    Test
    """
    def setUp(self):
        """

        Returns:

        """
        self.img_path = ""
        self.boxes1 = tf.convert_to_tensor([[[1.5, 1.5, 1, 1],
                                             [3.5, 1.5, 1, 1]],
                                            [[1.5, 3.5, 1., 1.],
                                             [3.5, 3.5, 1., 1.]]],
                                           dtype=tf.float32)
        self.boxes2 = tf.convert_to_tensor([[[1.5, 1.5, 1, 1],
                                             [4, 1, 1, 1]],
                                            [[1.5, 4.5, 1., 1.],
                                             [3, 4, 1., 1.]]],
                                           dtype=tf.float32)

        self.box_ious = tf.convert_to_tensor([[1, 1/7], [0, 1/7]],
                                             dtype=tf.float32)

    def test_iou(self):
        """

        Returns:

        """
        self.assertEqual(
            utils.bbox_iou_tf(self.boxes1, self.boxes2).numpy().tolist(),
            self.box_ious.numpy().tolist()
        )

    def test_img_resize(self):
        """

        Returns:

        """
        self.assertEqual(True, True)
        img_rd = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        tar_size = (320, 320)
        box = np.array([10, 10, 20, 20]).reshape((1, 4))
        _, bbox = utils.img_preprocess_fix_ratio(img_rd, tar_size, box)
        bbox_gt = np.array([45., 5., 50., 10.]).reshape((1, 4))

        self.assertEqual(bbox_gt.tolist(), bbox.tolist())


if __name__ == '__main__':
    unittest.main()
