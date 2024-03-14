import torch
import os
import util.data_pipeline
import util.image_visualizator
import util.annotation_utils
import util.image_utils

images_folder = "images/"
resources_path_train = "../resources/train/"
resources_path_val = "../resources/val/"

def check_cuda():
    print(torch.cuda.is_available())


def visualize_test_cases(folder_path):
    files = os.listdir(folder_path)

    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_name, image_extension = image_file.split('.')
        util.image_visualizator.image_visualize_xml(resources_path_val, images_folder, image_name, image_extension)


def multiply_test_cases(img_folder_path):
    util.data_pipeline.create_dataset_cropped_item(img_folder_path + "000001.png")


def convert_xml_to_json(folder_path):
    files = os.listdir(folder_path)

    annotation_files = [file for file in files if file.endswith(('.xml'))]
    for image_file in annotation_files:
        util.annotation_utils.convert_xml_to_json(folder_path + image_file)


if __name__ == "__main__":
    # visualize_test_cases(resources_path_val + images_folder)
    # convert_xml_to_json(resources_path_train)
    util.data_pipeline.create_dataset_item_v2(resources_path_val, images_folder, "000020.png")
    # util.annotation_utils.get_boxes_from_json(resources_path_val + "000020.json")


