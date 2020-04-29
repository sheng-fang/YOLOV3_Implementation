import numpy as np
import cv2

from core import utils


img_rd_1 = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
img_rd_2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
tar_size = (416, 416)
img_1, bbox_1 = utils.img_preprocess_fix_ratio(img_rd_1, tar_size)
img_2, bbox_2 = utils.img_preprocess_fix_ratio(img_rd_2, tar_size)
cv2.imshow("img 1", img_1.astype(np.uint8))
cv2.imshow("img 2", img_2.astype(np.uint8))
cv2.waitKey(0)