import json
import os

from PIL import Image, ImageDraw

from util import data_pipeline


def read_createml_json_file(json_file: str, class_map=None):
    with open(json_file) as json_obj:
        structure = json.load(json_obj)[0]

    bboxes = []
    labels = []

    for obj in structure['annotations']:
        xmin = obj['coordinates']['x'] - (obj['coordinates']['width'] / 2)
        ymin = obj['coordinates']['y'] - (obj['coordinates']['height'] / 2)
        xmax = obj['coordinates']['x'] + (obj['coordinates']['width'] / 2)
        ymax = obj['coordinates']['y'] + (obj['coordinates']['height'] / 2)
        bbox = [xmin, ymin, xmax, ymax]

        label = obj['label']
        try:
            label = int(label)
        except:
            label = int(class_map[label])

        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


def read_createml_json_dict(json_dict: dict, class_map=None):
    bboxes = []

    for obj in json_dict['annotations']:
        xmin = obj['coordinates']['x'] - (obj['coordinates']['width'] / 2)
        ymin = obj['coordinates']['y'] - (obj['coordinates']['height'] / 2)
        xmax = obj['coordinates']['x'] + (obj['coordinates']['width'] / 2)
        ymax = obj['coordinates']['y'] + (obj['coordinates']['height'] / 2)
        bbox = [xmin, ymin, xmax, ymax]

        bboxes.append(bbox)

    return bboxes


def draw_on_image(image, bboxes, color, width):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=width)
    return image


def visualise_image(image):
    image.show()


def draw_image_voc(image, items):
    draw = ImageDraw.Draw(image)
    for obj in items:
        print(obj['bbox'])
        xmin = obj['bbox'][0]
        ymin = obj['bbox'][1]
        xmax = obj['bbox'][2]
        ymax = obj['bbox'][3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()


def draw_image_bboxes(image, bboxes):
    draw = ImageDraw.Draw(image)
    print(image.width, image.height)
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()


def draw_image_bboxes_wh(image, bboxes):
    if isinstance(image, str):
        image = Image.open(image)
    draw = ImageDraw.Draw(image)
    print(image.width, image.height)
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()


def draw_image_json(image, json_dict):
    bboxes = read_createml_json_dict(json_dict)
    draw_image_bboxes(image, bboxes)


if __name__ == '__main__':
    img = Image.open('../dataset/images/0b9304e3-86e3-4d37-8d9f-962f33621d26.png')
    bboxes, labels = data_pipeline.read_createml_json('../dataset/train/0b9304e3-86e3-4d37-8d9f-962f33621d26.json')
    draw_image_bboxes(img, bboxes)
