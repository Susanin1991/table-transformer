import torch
import os

import data_pipeline
import util
import image_visualizator


images_folder = "images/"
resources_path = "../resources/"
detection_path = "detection/"
structure_path = "structure/"
train_path = "train/"
val_path = "val/"
experiment_path = "experiment/"

# resources_path_train_detection = "../resources/detection/train/"
# resources_path_val_detection = "../resources/detection/val/"
# resources_path_train_recognition = "../resources/structure/train/"
# resources_path_val_recognition = "../resources/structure/val/"
# resources_path_experiment = "../resources/experiment/"

def check_cuda():
    print(torch.cuda.is_available())


def visualize_test_cases(resources_path):
    files = os.listdir(os.path.join(resources_path, images_folder))

    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_name, image_extension = image_file.split('.')
        image_visualizator.image_visualize(resources_path, images_folder, image_name, image_extension)


def multiply_test_cases(resources_path, data_type_path, mode_path, images_folder):
    files = os.listdir(resources_path + data_type_path + mode_path + images_folder)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    for image in image_files:
        for _ in range(1):
            data_pipeline.create_augmented_detection_item_v2(resources_path, data_type_path, mode_path, images_folder, image)


def convert_xml_to_json(folder_path):
    files = os.listdir(folder_path)

    annotation_files = [file for file in files if file.endswith(('.xml'))]
    for image_file in annotation_files:
        util.annotation_utils.convert_xml_to_json(folder_path + image_file)


if __name__ == "__main__":
    multiply_test_cases(resources_path, experiment_path, val_path, images_folder)
    visualize_test_cases(resources_path + experiment_path + val_path)
    # convert_xml_to_json(resources_path_train)
    # util.data_pipeline.create_dataset_item_v3(resources_path_train)
    # util.annotation_utils.get_boxes_from_json(resources_path_val + "000020.json")
    # util.data_pipeline.create_augmented_detection_item_v2(resources_path_train)


