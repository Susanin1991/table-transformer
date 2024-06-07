import os

from PIL import Image, ImageDraw

import general_utils
import image_utils
import create_padded_dataset
import annotation_utils


def image_visualize_xml(resources_path, images_folder, image_name, image_extension):
    image = Image.open(resources_path + images_folder + image_name + "." + image_extension)
    draw = ImageDraw.Draw(image)

    # annotation = find_file_by_name(resources_path, image_no_extension)
    bboxes, labels, filename, width, height, database = create_padded_dataset.read_pascal_voc(resources_path + image_name + ".xml")
    # bboxes2 = util.annotation_utils.get_boxes_from_json(resources_path + image_name + ".json")
    # util.annotation_reader.read_xml(resources_path + image_name + ".xml")
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=8)
    image.show()


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
        xmax = bbox[2]
        ymax = bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=8)

    image.show()

# To get annotation path from image path
def get_annotation_path(image_path: str) -> str:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    directory = os.path.dirname(image_path)
    directory = directory.replace("\\images", "")
    res = ""
    for file_name in os.listdir(directory):
        if os.path.splitext(file_name)[0] == image_name:
            res = os.path.join(directory, file_name)
    return res

def get_bboxes_from_annotation(annotation_path: str) :
    if annotation_path.endswith('json'):
        class_map = general_utils.get_class_map('structure')
        bboxes, labels = image_utils.read_createml_json_file(annotation_path, class_map)
        print(f"Returning {len(bboxes)} bboxes and 4 None values")
        return bboxes, labels
    else:
        bboxes, labels, _, _, _, _ = create_padded_dataset.read_pascal_voc(annotation_path)
        return bboxes, labels


def image_visualize_old(resources_path, images_folder, image_name, image_extension):
    annotation_file_path = resources_path
    annotation_file = None
    files = os.listdir(annotation_file_path)
    for file in files:
        if file == image_name + ".json":
            print(".json")
            image_visualize_json(resources_path, images_folder, image_name, image_extension)
        if file == image_name + ".xml":
            print(".xml")
            image_visualize_xml(resources_path, images_folder, image_name, image_extension)


def image_visualize(args, image_path, bboxes, bboxes_modeled):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for box in bboxes:
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='green', width=8)

    for box in bboxes_modeled:
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=8)

    # image.show()
    image.save(args.save_images_path + image_path.split("\\")[-1])

