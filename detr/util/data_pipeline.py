import json
import multiprocessing
import os.path
import random
import shutil
import uuid
import sys

# import main
# from table_datasets import read_pascal_voc
sys.path.append(os.path.dirname("D:\\Work\\table-transformer\\detr\\util\\augmenter.py"))

from PIL import Image, ImageFilter
from tqdm import tqdm

import augmenter
import io_utils
import cv2util
import general_utils
import image_utils
import iou_utils
import textract_utils

dataset_path = './../resources'
source_path = './docs/original_dataset'
full_gt_path = './docs/full_ground_truth'
structure_dataset_path = './docs/structure_dataset'
structure_error_dataset = './docs/structure_error_dataset'
padding = 10


def clear_dataset():
    shutil.rmtree(dataset_path, ignore_errors=True)
    os.makedirs(dataset_path)
    img_path = os.path.join(dataset_path, 'images')
    print(img_path)
    os.makedirs(img_path, exist_ok=True)
    train_path = os.path.join(dataset_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    val_path = os.path.join(dataset_path, 'val')
    os.makedirs(val_path, exist_ok=True)
    test_path = os.path.join(dataset_path, 'test')
    os.makedirs(test_path, exist_ok=True)


def is_train_item():
    if random.randint(1, 100) <= 80:
        return True
    return False


def random_blur(image, max_blur):
    image = image.filter(filter=ImageFilter.GaussianBlur(radius=random.randint(0, max_blur) * 0.1))
    return image


def create_dataset_item(img_path):
    new_name = str(uuid.uuid4())
    new_img_path = os.path.join(dataset_path, 'images', new_name + '.png')
    if is_train_item():
        new_xml_path = os.path.join(dataset_path, 'train', new_name + '.xml')
    else:
        new_xml_path = os.path.join(dataset_path, 'val', new_name + '.xml')
    old_img_path = os.path.join(source_path, img_path)
    old_xml_name = img_path.split('.')[0]
    old_xml_path = os.path.join(source_path, old_xml_name + '.xml')
    image = Image.open(old_img_path).convert("RGB")
    image = random_blur(image, 40)
    image.save(new_img_path)
    shutil.copyfile(old_xml_path, new_xml_path)


def create_dataset_item_v3(resources_path, images_folder, image):
    new_name = str(uuid.uuid4())
    image_no_extension = image.split('.')[0]
    annotation = find_file_by_name(resources_path, image_no_extension)
    extension = '.xml'
    if annotation:
        extension = '.' + annotation.split('.')[1]
    new_img_path = os.path.join(resources_path, images_folder, new_name + '.png')
    new_annotation_path = os.path.join(resources_path, new_name + extension)
    old_annotation_path = os.path.join(resources_path, image_no_extension + extension)
    old_img_path = os.path.join(resources_path, images_folder, image)
    image = Image.open(old_img_path).convert("RGB")
    image = random_blur(image, 40)
    image.save(new_img_path)
    shutil.copyfile(old_annotation_path, new_annotation_path)


def find_file_by_name(resources_path, data_type_path, mode_path, image_no_extension):
    folder_path = os.path.join(resources_path, data_type_path, mode_path)
    files = os.listdir(folder_path)

    for file in files:
        if file.startswith(image_no_extension):
            return file

    return None


def create_dataset_item_v2(resources_path, img_folder, img_name):
    name = img_name.split('.')
    new_name = str(uuid.uuid4())
    new_img_path = resources_path + img_folder + new_name + '.png'
    new_annotation_path = resources_path + new_name + '.json'
    old_img_path = resources_path + img_folder + img_name
    old_annotation_path = resources_path + name[0] + '.json'
    image = Image.open(old_img_path).convert("RGB")
    image = random_blur(image, 40)
    image.save(new_img_path)
    with open(old_annotation_path) as json_obj:
        json_str = json_obj.read()
        structure = json.loads(json_str)
        formatted_json_str = json.dumps(structure, indent=2)
    with open(new_annotation_path, 'w') as json_file:
        json_file.write(formatted_json_str)
    # shutil.copyfile(old_annotation_path, new_annotation_path)

def create_dataset_cropped_item(img_path):
    new_name = str(uuid.uuid4())
    new_img_path = os.path.join(dataset_path, 'images', new_name + '.png')
    if is_train_item():
        new_xml_path = os.path.join(dataset_path, 'train', new_name + '.json')
    else:
        new_xml_path = os.path.join(dataset_path, 'val', new_name + '.json')
    old_img_path = os.path.join(structure_error_dataset, img_path)
    old_xml_name = img_path.split('.')[0]
    old_xml_path = os.path.join(structure_error_dataset, old_xml_name + '.json')
    image = Image.open(old_img_path).convert("RGB")
    image = random_blur(image, 20)
    image.save(new_img_path)
    shutil.copyfile(old_xml_path, new_xml_path)


def create_augmented_detection_item_v2(resources_path, data_type_path, mode_path, images_folder, image_name):
    new_name = str(uuid.uuid4())
    image_no_extension = image_name.split('.')[0]
    annotation = find_file_by_name(resources_path, data_type_path, mode_path, image_no_extension)
    extension = '.xml'
    if annotation:
        extension = '.' + annotation.split('.')[1]
    new_img_path = os.path.join(resources_path, data_type_path, mode_path, images_folder, new_name + '.png')
    new_annotation_path = os.path.join(resources_path, data_type_path, mode_path, new_name + extension)
    old_annotation_path = os.path.join(resources_path, data_type_path, mode_path, image_no_extension + extension)
    old_img_path = os.path.join(resources_path, data_type_path, mode_path, images_folder, image_name)
    image = Image.open(old_img_path).convert("RGB")

    if data_type_path == "detection/":
        class_map = general_utils.get_class_map('detection')
    else:
        class_map = general_utils.get_class_map('structure')

    if extension == '.xml':
        bboxes, labels = io_utils.read_pascal_voc(old_annotation_path, class_map)
    else:
        bboxes, labels = image_utils.read_createml_json_file(old_annotation_path, class_map)

    t_bboxes = []
    t_labels = []
    for i in range(len(bboxes)):
        if labels[i] == 0 or labels[i] == 5 or labels[i] == 'table':
            t_bboxes.append(bboxes[i])
            t_labels.append(labels[i])
    image, t_bboxes, t_labels = augmenter.augment(image, t_bboxes, t_labels)
    image.save(new_img_path)
    if extension == '.xml':
        tree = io_utils.modify_and_save_xml(old_annotation_path, new_annotation_path, t_bboxes)
        tree.write(new_annotation_path, encoding="utf-8", xml_declaration=True)
    else:
        json_dict = io_utils.write_cml_json(image_no_extension + extension, t_bboxes, t_labels, class_map)
        io_utils.save_json_full_path(new_annotation_path, [json_dict])


def create_augmented_detection_item(img_name):
    new_name = str(uuid.uuid4())
    new_img_path = os.path.join(dataset_path, 'images', f'{new_name}.png')
    if is_train_item():
        new_json_path = os.path.join(dataset_path, 'train', f'{new_name}.json')
    else:
        new_json_path = os.path.join(dataset_path, 'val', f'{new_name}.json')
    old_img_path = os.path.join(source_path, img_name)
    old_json_name = img_name.split('.')[0]
    old_json_path = os.path.join(full_gt_path, f'{old_json_name}.json')
    image = Image.open(old_img_path).convert("RGB")
    bboxes, labels = io_utils.read_cml_json_file(old_json_path, general_utils.get_class_map('structure'))
    t_bboxes = []
    t_labels = []
    for i in range(len(bboxes)):
        if labels[i] == 0 or labels[i] == 'table':
            t_bboxes.append(bboxes[i])
            t_labels.append(labels[i])
    image, t_bboxes, t_labels = augmenter.augment(image, t_bboxes, t_labels)
    image.save(new_img_path)
    json_dict = io_utils.write_cml_json(img_name, t_bboxes, t_labels, general_utils.get_class_map('detection'))
    io_utils.save_json_full_path(new_json_path, [json_dict])


def create_augmented_test_item(img_name):
    new_name = str(uuid.uuid4())
    new_img_path = os.path.join(dataset_path, 'images', f'{new_name}.png')
    new_json_path = os.path.join(dataset_path, 'test', f'{new_name}.json')
    old_img_path = os.path.join(source_path, img_name)
    old_json_name = img_name.split('.')[0]
    old_json_path = os.path.join(full_gt_path, f'{old_json_name}.json')
    image = Image.open(old_img_path).convert("RGB")
    bboxes, labels = io_utils.read_cml_json_file(old_json_path, general_utils.get_class_map('structure'))
    image, bboxes, labels = augmenter.augment(image, bboxes, labels)
    image.save(new_img_path)
    json_dict = io_utils.write_cml_json(img_name, bboxes, labels, general_utils.get_class_map('structure'))
    io_utils.save_json_full_path(new_json_path, [json_dict])


def make_img_list(path, amount):
    old_list = os.listdir(path)
    new_list = []
    for img in old_list:
        if img.endswith('.png'):
            for i in range(amount):
                new_list.append(img)
    print(f'dataset size: {len(new_list)}')
    return new_list


def crop_to_single_tables(img_name):
    i = 0
    full_img_path = os.path.join(source_path, img_name)
    img = Image.open(full_img_path)
    file_name = img_name.split('.')[0]
    full_xml_path = os.path.join(source_path, file_name + '.json')
    bboxes, labels = io_utils.read_pascal_voc(full_xml_path, general_utils.get_class_map('detection'))
    for bbox in bboxes:
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
        cropped_img = img.crop(bbox)
        new_img_path = os.path.join(structure_dataset_path, f'{file_name}_{i}.png')
        cropped_img.save(new_img_path)
        i += 1


def crop_table_with_shift(file_name, index, bbox):
    x_error = random.randint(-20, 20)
    y_error = random.randint(-20, 20)
    json_path = os.path.join(structure_dataset_path, f'{file_name}_{index}.json')
    bbox = [bbox[0] + x_error, bbox[1] + y_error, bbox[2] + x_error, bbox[3] + y_error]
    bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
    img_path = os.path.join(source_path, f'{file_name}.png')
    img = Image.open(img_path)
    img = img.crop(bbox)
    new_img_path = os.path.join(structure_error_dataset, f'{file_name}_{index}.png')
    img.save(new_img_path)
    json_file = open(json_path)
    json_dict = json.load(json_file)[0]
    new_json = move_cords(json_dict, file_name, index, x_error, y_error, img.width, img.height)
    if index == 0:
        image_utils.draw_image_json(img, new_json)


def crop_document_with_shift(img_name):
    i = 0
    file_name = img_name.split('.')[0]
    full_xml_path = os.path.join(source_path, file_name + '.xml')
    bboxes, labels = io_utils.read_pascal_voc(full_xml_path, general_utils.get_class_map('detection'))
    for bbox in bboxes:
        crop_table_with_shift(file_name, i, bbox)
        i += 1


def shift_cords(bbox, x_shift, y_shift, width, height):
    xmin = bbox[0] - x_shift
    xmin = 0 if xmin < 0 else xmin
    ymin = bbox[1] - y_shift
    ymin = 0 if ymin < 0 else ymin
    xmax = bbox[2] - x_shift
    xmax = xmax if xmax <= width else width
    ymax = bbox[3] - y_shift
    ymax = ymax if ymax <= height else height
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def move_cords(json_dict, file_name, index, x_shift, y_shift, width, height):
    json_dict['image'] = f'{file_name}_{index}.png'
    for obj in json_dict['annotations']:
        bbox = io_utils.cml_coordinates_to_bbox(obj['coordinates'])
        bbox = shift_cords(bbox, x_shift, y_shift, width, height)
        obj['coordinates'] = io_utils.bbox_to_cml_cords(bbox)
    with open(os.path.join(structure_error_dataset, f'{file_name}_{index}.json'), 'w+') as out_file:
        json.dump([json_dict], out_file)
    return json_dict


def move_cords_v2(doc_dict, crop_dict, x_shift, y_shift, width, height):
    for obj in crop_dict['annotations']:
        bbox = io_utils.cml_coordinates_to_bbox(obj['coordinates'])
        bbox = shift_cords(bbox, x_shift, y_shift, width, height)
        obj['coordinates'] = io_utils.bbox_to_cml_cords(bbox)
        doc_dict['annotations'].append(obj)
    return doc_dict


def create_structure_with_cv2(img_name):
    cords = cv2util.get_cells(structure_dataset_path, img_name, denoise=False, thresh=180, padding=(23, 10, 22, 10))
    n_cords = cv2util.exclude_bboxes(cords)
    final = n_cords
    json_dict = io_utils.cv2cords_to_cml(img_name, final)
    io_utils.save_json('./docs/cv2_prediction', img_name, [json_dict])


def move_crop_cords(image: Image, bbox: list, doc_dict: dict, crop_dict: dict, padding=10):
    x_shift = -bbox[0]+padding
    y_shift = -bbox[1]+padding
    w = image.width
    h = image.height
    doc_dict = move_cords_v2(doc_dict, crop_dict, x_shift, y_shift, w, h)
    return doc_dict


def combine_gt(img_name: str):
    class_map = general_utils.get_class_map('detection')
    file = img_name.split('.')[0] + '.xml'
    bboxes, labels = io_utils.read_pascal_voc(os.path.join(source_path, file), class_map)
    json_dict = io_utils.write_cml_json(img_name, [], [], class_map)
    img = Image.open(os.path.join(source_path, img_name))
    for i in range(len(bboxes)):
        crop_file = os.path.join(structure_dataset_path, f'{file.split(".")[0]}_{i}.json')
        crop_dict = io_utils.open_cml_json(crop_file)
        json_dict = move_crop_cords(img, bboxes[i], json_dict, crop_dict)
        io_utils.save_json(source_path, img_name, [json_dict])


def combine_cv(img_name: str):
    class_map = general_utils.get_class_map('detection')
    file = img_name.split('.')[0] + '.xml'
    bboxes, labels = io_utils.read_pascal_voc(os.path.join('./docs/original_xml', file), class_map)
    json_dict = io_utils.write_cml_json(img_name, [], [], class_map)
    img = Image.open(os.path.join(source_path, img_name))
    for i in range(len(bboxes)):
        crop_file = os.path.join('./docs/cv2_prediction', f'{file.split(".")[0]}_{i}.json')
        crop_dict = io_utils.open_cml_json(crop_file)
        json_dict = move_crop_cords(img, bboxes[i], json_dict, crop_dict)
        io_utils.save_json('./docs/cv2_doc_prediction', img_name, [json_dict])


def merge_gt_blanks(img_name: str):
    class_map = general_utils.get_class_map('structure')
    file = img_name.split('.')[0]
    img = Image.open(os.path.join(source_path, img_name))
    w = img.width
    h = img.height
    bboxes_tb, labels_tb = io_utils.read_cml_json_file(os.path.join('./docs/original_xml', f'{file}.xml'),
                                                       general_utils.get_class_map('detection'))
    bboxes_gt, labels_gt = io_utils.read_cml_json_file(os.path.join('./docs/original_dataset', f'{file}.json'),
                                                       class_map)
    bboxes_bl, labels_bl = io_utils.read_cml_json_file(os.path.join('./docs/blanks', f'{file}.json'),
                                                       class_map)
    ocr = textract_utils.load_ocr('./docs/document_ocr', f'{file}.json')
    bboxes_ocr, words_ocr = textract_utils.ocr_to_words(ocr, w, h)
    labels_ocr = [8 for i in bboxes_ocr]
    bboxes_tb.extend(bboxes_gt)
    bboxes_tb.extend(bboxes_bl)
    bboxes_tb.extend(bboxes_ocr)
    labels_tb.extend(labels_gt)
    labels_tb.extend(labels_bl)
    labels_tb.extend(labels_ocr)
    json_dict = io_utils.write_cml_json(img_name, bboxes_tb, labels_tb, class_map)
    io_utils.save_json('./docs/full_ground_truth', img_name, [json_dict])


def get_metrics():
    metrics = iou_utils.metrics_from_cv2('./docs/cv2_doc_prediction', source_path, iou_thresh=0.7)
    print()
    for i in range(len(metrics) - 1):
        file, prec, rec, p_bb = metrics[i]
        if not file.endswith('.png'):
            file = file.split('.')[0] + '.png'
        if prec < 0.7:
            print(f'{file}: {prec:.2f} {rec:.2f}')
            cv2util.compare_with_gt_bbf('./docs/original_dataset', file, p_bb)


def generate_ocr(img_name: str):
    ocr = textract_utils.get_text_textract('./docs/original_dataset', img_name)
    io_utils.save_json('./docs/document_ocr', img_name, ocr)


def generate_blanks_json(img_name: str):
    file_name = img_name.split('.')[0]
    ocr = textract_utils.load_ocr('./docs/document_ocr', file_name + '.json')
    img = Image.open(os.path.join('./docs/original_dataset', img_name))
    w, h = img.width, img.height
    cluster_diff = h * 0.007
    ocr_bboxes, ocr_words = textract_utils.ocr_to_words(ocr, w, h)
    ocr_list = textract_utils.words_n_bboxes_to_inner_str(ocr_words, ocr_bboxes)
    bboxes, labels = io_utils.read_cml_json_file(f'./docs/original_dataset/{file_name}.json',
                                                 general_utils.get_class_map('structure'))
    blank_cells = textract_utils.find_blanks(bboxes, ocr_list, cluster_diff)
    blank_cords = io_utils.bboxes_to_cml_cords(blank_cells)
    blanks_json = io_utils.build_cml_json(img_name, blank_cords, label='blank')
    io_utils.save_json('./docs/blanks', img_name, [blanks_json])


def show_dataset_item(img_name: str):
    file_name = img_name.replace('.png', '.json')
    img_path = os.path.join(dataset_path, 'images')
    img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
    if os.path.exists(os.path.join(dataset_path, 'train', file_name)):
        annot_path = os.path.join(os.path.join(dataset_path, 'train'))
    else:
        annot_path = os.path.join(os.path.join(dataset_path, 'val'))
    bboxes, labels = io_utils.read_cml_json_file(os.path.join(annot_path, file_name),
                                                 general_utils.get_class_map('structure'))
    table_bboxes = []
    cell_bboxes = []
    blank_bboxes = []
    word_bboxes = []
    for i in range(len(bboxes)):
        if labels[i] == 'table' or labels[i] == 0:
            table_bboxes.append(bboxes[i])
        if labels[i] == 'table spanning cell' or labels[i] == 5:
            cell_bboxes.append(bboxes[i])
        if labels[i] == 'blank' or labels[i] == 7:
            blank_bboxes.append(bboxes[i])
        if labels[i] == 'word' or labels[i] == 8:
            word_bboxes.append(bboxes[i])
    image_utils.draw_on_image(img, table_bboxes, 'purple', 2)
    image_utils.draw_on_image(img, cell_bboxes, 'green', 2)
    image_utils.draw_on_image(img, blank_bboxes, 'red', 3)
    image_utils.draw_on_image(img, word_bboxes, 'blue', 1)
    image_utils.visualise_image(img)


if __name__ == '__main__':
    # clear_dataset()
    # items = make_img_list(source_path, 87)
    # general_utils.run_multiprocessing(create_augmented_detection_item, items)
    show_dataset_item('0ade75b5-316e-481a-9110-5ee9b976bdda.png')

    # pass
