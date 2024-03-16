from PIL import Image, ImageDraw
import scripts.create_padded_dataset
import util.annotation_utils


def image_visualize_xml(resources_path, images_folder, image_name, image_extension):
    image = Image.open(resources_path + images_folder + image_name + "." + image_extension)
    draw = ImageDraw.Draw(image)
    bboxes, labels, filename, width, height, database = scripts.create_padded_dataset.read_pascal_voc(resources_path + image_name + ".xml")
    # bboxes2 = util.annotation_utils.get_boxes_from_json(resources_path + image_name + ".json")
    # util.annotation_reader.read_xml(resources_path + image_name + ".xml")
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()


def image_visualize_json(resources_path, images_folder, image_name, image_extension):
    image = Image.open(resources_path + images_folder + image_name + "." + image_extension)
    draw = ImageDraw.Draw(image)
    bboxes, labels, filename, width, height, database = scripts.create_padded_dataset.read_pascal_voc(resources_path + image_name + ".json")
    # util.annotation_reader.read_xml(resources_path + image_name + ".xml")
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()

