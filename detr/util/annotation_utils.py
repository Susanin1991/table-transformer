import xmltodict
import xml.etree.ElementTree as ET
import json
import os


def convert_xml_to_json(xml_file_path: str):
    file_name = os.path.splitext(xml_file_path)[0]
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data_dict = element_to_dict(root)
    json_str = json.dumps(data_dict, indent=2)

    with open(file_name + '.json', 'w') as json_file:
        json_file.write(json_str)


def element_to_dict(element):
    result = {}
    for child in element:
        if child.tag in result:
            if isinstance(result[child.tag], list):
                result[child.tag].append(element_to_dict(child))
            else:
                result[child.tag] = [result[child.tag], element_to_dict(child)]
        else:
            result[child.tag] = element_to_dict(child) if len(child) else child.text
    return result


def save_json(json_string, xml_file):
    file_name = os.path.splitext(xml_file)[0]

    json_file = file_name + '.json'

    with open(json_file, 'w') as f:
        f.write(json_string)


def get_boxes_from_json(json_file: str, class_map=None):
    with open(json_file) as json_obj:
        json_str = json_obj.read()
        structure = json.loads(json_str)

    bboxes = []
    labels = []
    if type(structure['object']) == dict:
        xmin = float(structure['object']['bndbox']['xmin'])
        ymin = float(structure['object']['bndbox']['ymin'])
        xmax = float(structure['object']['bndbox']['xmax'])
        ymax = float(structure['object']['bndbox']['ymax'])
        bbox = [xmin, ymin, xmax, ymax]
        bboxes.append(bbox)
    else:
        for obj in structure['object']:
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)

    return bboxes
