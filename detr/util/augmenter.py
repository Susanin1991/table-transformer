import os

import albumentations as A
import cv2
import numpy as np
from PIL import Image

import general_utils
import io_utils
import image_utils


def zoom():
    return A.ShiftScaleRotate(shift_limit=0,
                              scale_limit=0.1,
                              rotate_limit=0,
                              p=0.3,
                              border_mode=cv2.BORDER_REPLICATE)


def shift():
    return A.ShiftScaleRotate(shift_limit=0.01,
                              scale_limit=0,
                              rotate_limit=0,
                              p=1,
                              border_mode=cv2.BORDER_REPLICATE)


def rotate():
    return A.ShiftScaleRotate(shift_limit=0,
                              scale_limit=0,
                              rotate_limit=1,
                              p=0.1,
                              border_mode=cv2.BORDER_REPLICATE)


def contrast():
    return A.Compose([A.RandomBrightnessContrast(brightness_limit=0.5,
                                                 contrast_limit=0.1,
                                                 p=1),
                      A.CLAHE(p=1)], p=0.2)


def blur():
    return A.Blur(blur_limit=(3, 4), p=0.2)


def noise():
    return A.GaussNoise(var_limit=(0, 50), p=0.2)


def compression():
    return A.ImageCompression(quality_lower=90,
                              quality_upper=99,
                              p=0.3)


def augment(image: Image, bboxes: list, labels: list):
    num_img = np.array(image)
    transform = A.Compose([zoom(), shift(), rotate(), contrast(), blur(), noise(), compression()],
                          bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    augmented = transform(image=num_img, bboxes=bboxes, labels=labels)
    bboxes = augmented['bboxes']
    num_img = augmented['image']
    labels = augmented['labels']
    image = Image.fromarray(np.uint8(num_img))
    return image, bboxes, labels


if __name__ == '__main__':
    file_name = '000002'
    table_bboxes = []
    cell_bboxes = []
    blank_bboxes = []
    word_bboxes = []
    img = Image.open(os.path.join('./docs/original_dataset', file_name + '.png'))
    bboxes, labels = io_utils.read_cml_json_file(f'./docs/full_ground_truth/{file_name}.json',
                                                 general_utils.get_class_map('structure'))
    new_img, new_bboxes, new_labels = augment(img, bboxes, labels)
    for i in range(len(new_bboxes)):
        if new_labels[i] == 'table' or new_labels[i] == 0:
            table_bboxes.append(new_bboxes[i])
        if new_labels[i] == 'table spanning cell' or new_labels[i] == 5:
            cell_bboxes.append(new_bboxes[i])
        if new_labels[i] == 'blank' or new_labels[i] == 7:
            blank_bboxes.append(new_bboxes[i])
        if new_labels[i] == 'word' or new_labels[i] == 8:
            word_bboxes.append(new_bboxes[i])
    image_utils.draw_on_image(new_img, table_bboxes, 'yellow', 1)
    image_utils.draw_on_image(new_img, cell_bboxes, 'green', 2)
    image_utils.draw_on_image(new_img, blank_bboxes, 'red', 3)
    image_utils.draw_on_image(new_img, word_bboxes, 'blue', 1)
    image_utils.visualise_image(new_img)
