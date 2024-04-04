import os

from PIL import Image, ImageDraw

import general_utils
import image_utils
import scripts.create_padded_dataset
import annotation_utils


def image_visualize_xml(resources_path, images_folder, image_name, image_extension):
    image = Image.open(resources_path + images_folder + image_name + "." + image_extension)
    draw = ImageDraw.Draw(image)

    # annotation = find_file_by_name(resources_path, image_no_extension)
    bboxes, labels, filename, width, height, database = scripts.create_padded_dataset.read_pascal_voc(resources_path + image_name + ".xml")
    # bboxes2 = util.annotation_utils.get_boxes_from_json(resources_path + image_name + ".json")
    # util.annotation_reader.read_xml(resources_path + image_name + ".xml")
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()


def image_visualize(resources_path, images_folder, image_name, image_extension):
    annotation_file_path = resources_path
    annotation_file = None
    files = os.listdir(annotation_file_path)
    for file in files:
        if file.startswith(image_name):
            annotation_file = file
    if annotation_file.endswith(".json"):
        image_visualize_json(resources_path, images_folder, image_name, image_extension)
    else:
        image_visualize_xml(resources_path, images_folder, image_name, image_extension)


def image_visualize_json(resources_path, images_folder, image_name, image_extension):
    image = Image.open(resources_path + images_folder + image_name + "." + image_extension)
    draw = ImageDraw.Draw(image)
    class_map = general_utils.get_class_map('structure')
    bboxes, labels = image_utils.read_createml_json_file(resources_path + image_name + ".json", class_map)
    # bboxes, labels, filename, width, height, database = scripts.create_padded_dataset.read_pascal_voc(resources_path + image_name + ".json")
    # util.annotation_reader.read_xml(resources_path + image_name + ".xml")
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()

