import json
import os.path

import xml.etree.ElementTree as ET


def read_pascal_voc(xml_file: str, class_map=None):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None

        label = object_.find("name").text
        try:
            label = int(label)
        except:
            label = int(class_map[label])

        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)
        bbox = [xmin, ymin, xmax, ymax]  # PASCAL VOC

        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


def read_cml_json_file(json_file: str, class_map=None):
    with open(json_file) as json_obj:
        structure = json.load(json_obj)[0]
    bboxes = []
    labels = []

    for obj in structure['annotations']:
        bbox = cml_coordinates_to_bbox(obj['coordinates'])

        label = obj['label']
        try:
            label = int(label)
        except:
            label = int(class_map[label])

        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


def read_cml_json_dict(json_dict: dict, class_map=None):
    bboxes = []
    labels = []

    for obj in json_dict['annotations']:
        bbox = cml_coordinates_to_bbox(obj['coordinates'])

        label = obj['label']
        try:
            label = int(label)
        except:
            label = int(class_map[label])

        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


def open_cml_json(json_file: str) -> dict:
    with open(json_file) as json_obj:
        structure = json.load(json_obj)[0]
    return structure


def write_cml_json(image_name: str, bboxes: list, labels: list, class_map: dict) -> dict:
    json_dict = {'image': image_name, 'annotations': []}
    if len(bboxes) == 0:
        return json_dict
    for i in range(len(bboxes)):
        label = labels[i]
        if isinstance(labels[i], int):
            label = list(class_map.keys())[list(class_map.values()).index(labels[i])]
        obj = {'label': label, 'coordinates': bbox_to_cml_cords(bboxes[i])}
        json_dict['annotations'].append(obj)
    return json_dict


def objects_to_cml(img_name, objects):
    json_dict = {}
    json_dict['image'] = img_name
    json_dict['annotations'] = []
    for obj in objects['objects']:
        label = obj['label']
        xmin = obj['bbox'][0]
        ymin = obj['bbox'][1]
        width = obj['bbox'][2] - xmin
        height = obj['bbox'][3] - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        dict_obj = {'label': label, 'coordinates': {'x': x, 'y': y, 'width': width, 'height': height}}
        json_dict['annotations'].append(dict_obj)
    return json_dict


def cv2cords_to_cml(img_name, coordinates, label='table spanning cell'):
    # cv2 coordinates: xmin, ymin, width, height
    # createml coordinates: x(center), y(center), width, height
    json_dict = {}
    json_dict['image'] = img_name
    json_dict['annotations'] = []
    for cord in coordinates:
        label = label
        xmin, ymin, width, height = cord
        xmin -= 4
        ymin -= 4
        width += 8
        height += 8
        x = xmin + width / 2
        y = ymin + height / 2
        dict_obj = {'label': label, 'coordinates': {'x': x, 'y': y, 'width': width, 'height': height}}
        json_dict['annotations'].append(dict_obj)
    return json_dict


def build_cml_json(img_name, cords, label='table spanning cell'):
    json_dict = {'image': img_name, 'annotations': []}
    for cord in cords:
        label = label
        dict_obj = {'label': label, 'coordinates': cord}
        json_dict['annotations'].append(dict_obj)
    return json_dict


def cml_coordinates_to_bbox(coordinates):
    xmin = coordinates['x'] - (coordinates['width'] / 2)
    ymin = coordinates['y'] - (coordinates['height'] / 2)
    xmax = coordinates['x'] + (coordinates['width'] / 2)
    ymax = coordinates['y'] + (coordinates['height'] / 2)
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def bbox_to_cml_cords(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = bbox[0] + width / 2
    y = bbox[1] + height / 2
    cords = {'x': x, 'y': y, 'width': width, 'height': height}
    return cords


def bboxes_to_cml_cords(bboxes):
    cords = []
    for bbox in bboxes:
        cord = bbox_to_cml_cords(bbox)
        cords.append(cord)
    return cords


def save_json(folder, img_name: str, json_dict):
    json_file_name = os.path.join(folder, img_name.split('.')[0] + '.json')
    with open(json_file_name, 'w+') as out_file:
        json.dump(json_dict, out_file)


def save_json_full_path(path, json_dict):
    with open(path, 'w+') as out_file:
        json.dump(json_dict, out_file)


def modify_and_save_xml(input_file, output_file, bboxes):
    tree = ET.parse(input_file)
    root = tree.getroot()

    filename = os.path.basename(input_file)
    path_element = root.find("path")
    path_element.text = path_element.text.replace(filename, output_file)

    for obj, bbox in zip(root.findall("object"), bboxes):
        bndbox = obj.find("bndbox")
        bndbox.find("xmin").text = str(bbox[0])
        bndbox.find("xmax").text = str(bbox[1])
        bndbox.find("ymin").text = str(bbox[2])
        bndbox.find("ymax").text = str(bbox[3])

    return tree