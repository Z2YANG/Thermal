import cv2 as cv
import numpy as np
from mrcnn import config
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.visualize import display_instances


# 若有多个mask，则将其整合到一张图片中，mask部分设置为黑色
def masks_integration(masks, img):
    row, col, num = masks.shape
    new_image = np.zeros(img.shape, dtype=np.uint8)
    for r in range(row):
        for c in range(col):
            for n in range(num):
                if masks[r, c, n] == True:
                    new_image[r, c] = 0
                    break
                else:
                    new_image[r, c] = img[r, c]
    return new_image


# 根据mask，计算人脸的几何中心
def calculate_center(masks):
    row, col, num = masks.shape
    cnt = 0
    sum = np.zeros(2, dtype=np.int32)
    for r in range(row):
        for c in range(col):
            for n in range(num):
                if masks[r, c, n] == True:
                    sum += np.array([r, c])
                    cnt += 1
    return np.array([i // cnt for i in sum])


# 围绕中心旋转图片并计算最佳旋转角，使得治疗前后图片mask尽可能重合
def calculate_rotation_angle(masks_pretreat, masks_posttreat, img_pretreat,
                             img_posttreat):
    # 计算面部几何中心，并算出治疗前后几何中心的偏移量
    center_pretreat = calculate_center(masks_pretreat)
    center_posttreat = calculate_center(masks_posttreat)
    center_diff = center_posttreat - center_pretreat

    # 带有mask的图片，用于旋转
    masked_img_pretreat = masks_integration(masks_pretreat, img_pretreat)
    masked_img_posttreat = masks_integration(masks_posttreat, img_posttreat)

    row, col = img_posttreat.shape[:2]
    max_overlap = 0
    rotation_angle = 0
    masks_pretreat_area = 0  # 治疗前图片mask的面积
    masks_posttreat_area = 0  #治疗后mask的面积

    # 计算masks的面积
    for r in range(row):
        for c in range(col):
            if masked_img_pretreat[r, c, 0] == 0:
                masks_pretreat_area += 1
            if masked_img_posttreat[r, c, 0] == 0:
                masks_posttreat_area += 1

    # 平移治疗后的图片，使治疗前后的图片中心重合
    M = np.float32([[1, 0, -center_diff[1]], [0, 1, -center_diff[0]]])
    translated_masked_img_posttreat = cv.warpAffine(
        masked_img_posttreat, M, (col, row), borderMode=cv.BORDER_REPLICATE)

    # 旋转治疗后的图片，每次旋转1°，从-15°至15°
    for ag in range(30):
        rotate = cv.getRotationMatrix2D(
            (int(center_posttreat[1]), int(center_posttreat[0])), ag - 15, 1)
        rotated_trans_masked_img_posttreat = cv.warpAffine(
            translated_masked_img_posttreat,
            rotate, (col, row),
            borderMode=cv.BORDER_REPLICATE)
        overlap = 0

        # 计算图片平移后的mask重合面积
        for r in range(row):
            for c in range(col):
                if masked_img_pretreat[
                        r, c,
                        0] == 0 and rotated_trans_masked_img_posttreat[r, c,
                                                                       0] == 0:
                    overlap += 1

        if overlap > max_overlap:
            max_overlap = overlap
            rotation_angle = ag - 15

    rt_display = cv.getRotationMatrix2D(
        (int(center_posttreat[1]), int(center_posttreat[0])), rotation_angle,
        1)
    rt_img_display = cv.warpAffine(translated_masked_img_posttreat,
                                   rt_display, (col, row),
                                   borderMode=cv.BORDER_REPLICATE)

    masked_img_pretreat[center_pretreat[0], center_pretreat[1]] = 255
    masked_img_posttreat[center_posttreat[0], center_posttreat[1]] = 255
    rt_img_display[center_pretreat[0], center_pretreat[1]] = 255

    print(center_posttreat)
    print(center_diff)
    print(masks_pretreat_area, masks_posttreat_area, max_overlap)
    cv.imshow('masked_img_pretreat', masked_img_pretreat)
    cv.imshow('masked_img_posttreat', masked_img_posttreat)
    cv.imshow('rt_img_display', rt_img_display)
    return rotation_angle  # 正值为逆时针，负值为顺时针


class FaceConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "face"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # IMAGE_RESIZE_MODE = "none"
    # IMAGE_MIN_DIM = 400
    # IMAGE_MAX_DIM = 512

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + face

    # Number of training steps per epoch

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


src_pretreat = cv.imread('thermal_picture/b1.png')
src_posttreat = cv.imread('thermal_picture/a1.png')

config = FaceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs')
model.load_weights('logs/mask_rcnn_face_0030.h5', by_name=True)

result_pretreat = model.detect([src_pretreat])
result_posttreat = model.detect([src_posttreat])

masks_pretreat = result_pretreat[0]['masks']
masks_posttreat = result_posttreat[0]['masks']

optimal_rotation_angle = calculate_rotation_angle(masks_pretreat,
                                                  masks_posttreat,
                                                  src_pretreat, src_posttreat)
print(optimal_rotation_angle)

cv.waitKey(0)
cv.destroyAllWindows()
'''
class_names = ['BG', 'face']
image = cv.imread('images/val/111.png')
result = model.detect([image])
display_instances(image,
                  result[0]['rois'],
                  result[0]['masks'],
                  result[0]['class_ids'],
                  class_names,
                  scores=result[0]['scores'],
                  title="",
                  figsize=(16, 16),
                  ax=None,
                  show_mask=True,
                  show_bbox=True,
                  colors=None,
                  captions=None)
'''
