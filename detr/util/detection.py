import json
import os
from pprint import pprint

import general_utils
import image_utils
import io_utils
import cv2_prediction
import data_pipeline
import inference
from PIL import Image, ImageDraw

import iou_utils
import textract_utils
from image_utils import draw_on_image

PARAMETER = 7
DATA_DIR = './dataset'
PADDING = 10


det_config_path = os.path.abspath('../../src/structure_config.json')
det_model_path = os.path.abspath('../../resources/detection/output/model_20.pth')
str_config_path = os.path.abspath('../../src/structure_config.json')
str_model_path = os.path.abspath('../../resources/structure/output/model_20.pth')
det_model_path_base = os.path.abspath('../../resources/detection/output/base.pth')
str_model_path_base = os.path.abspath('../../resources/structure/output/base.pth')


def draw_image(image, items):
    draw = ImageDraw.Draw(image)
    for obj in items:
        print(obj['bbox'])
        xmin = obj['bbox'][0]
        ymin = obj['bbox'][1]
        xmax = obj['bbox'][2]
        ymax = obj['bbox'][3]
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
    image.show()


def draw_image_with_bbox(image, bboxes, color):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=3)
    image.show()


pipe = inference.TableExtractionPipeline(det_config_path=det_config_path,
                                         det_model_path=det_model_path_base,
                                         det_device='cuda',
                                         str_config_path=str_config_path,
                                         str_model_path=str_model_path_base,
                                         str_device='cuda')


def detect_tables(img_dir):
    images = os.listdir(img_dir)
    for img in images:
        if img.endswith('.png'):
            img_path = os.path.join(img_dir, img)
            print(img_path)
            image = Image.open(img_path).convert("RGB")
            outputs = pipe.detect(image)
            print(outputs)
            draw_image(image, outputs['objects'])


def recognise_cells(img_dir):
    print(img_dir)
    images = os.listdir(img_dir)
    for img in images:
        if img.endswith('.png'):
            img_path = os.path.join(img_dir, img)
            print(img_path)
            image = Image.open(img_path).convert("RGB")
            outputs = pipe.recognize(image, out_objects=True)
            draw_image(image, outputs['objects'])


def recognise_cells_v2(image, outputs, padding=10):
    crops = []
    table_bboxes = []
    str_outs = []
    for obj in outputs['objects']:
        crop_bbox = [obj['bbox'][0] - padding,
                     obj['bbox'][1] - padding,
                     obj['bbox'][2] + padding,
                     obj['bbox'][3] + padding]
        cropped_img = image.crop(crop_bbox)
        crops.append(cropped_img)
        table_bboxes.append(obj['bbox'])
    for crop in crops:
        str_out = pipe.recognize(crop, out_objects=True)
        str_outs.append(str_out)
    return str_outs, table_bboxes


def extract_table_structure(img_dir):
    images = os.listdir(img_dir)
    outputs = []
    for img in images:
        if img.endswith('.png'):
            img_path = os.path.join(img_dir, img)
            image = Image.open(img_path).convert("RGB")
            outputs = pipe.extract(image, crop_padding=10)
            # for output in outputs:
            #     draw_image(output['image'], output['objects'])
    return outputs


def one_table_predictions(image, padding=10):
    p_bboxes = []
    p_labels = []
    outputs = pipe.detect(image)
    str_outs, table_bboxes = recognise_cells_v2(image, outputs, padding)
    for i in range(len(table_bboxes)):
        x_shift = -table_bboxes[i][0] + padding
        y_shift = -table_bboxes[i][1] + padding
        w = image.width
        h = image.height
        for str_out in str_outs:
            for i in range(len(str_out['objects'])):
                bbox = str_out['objects'][i]['bbox']
                bbox = data_pipeline.shift_cords(bbox, x_shift, y_shift, w, h)
                label = str_out['objects'][i]['label']
                p_bboxes.append(bbox)
                p_labels.append(label)
    return p_bboxes, p_labels


# added function
def detect_page_cells(image):
    outputs = pipe.detect(image)
    tables = []
    table_bboxes = []
    for obj in outputs['objects']:
        crop_bbox = [obj['bbox'][0] - PADDING,
                     obj['bbox'][1] - PADDING,
                     obj['bbox'][2] + PADDING,
                     obj['bbox'][3] + PADDING]
        cropped_img = image.crop(crop_bbox)
        tables.append(cropped_img)
        table_bboxes.append(obj['bbox'])

    # image = get_img_preview(image, table_bboxes, 'green', 3)
    # image.show()

    cells = []
    for i in range(len(tables)):
        str_outs = pipe.recognize(tables[i], out_objects=True)
        x_shift = -table_bboxes[i][0] + PADDING
        y_shift = -table_bboxes[i][1] + PADDING
        w = image.width
        h = image.height
        for str_obj in str_outs['objects']:
            bbox = str_obj['bbox']
            bbox = data_pipeline.shift_cords(bbox, x_shift, y_shift, w, h)
            cells.append(bbox)
    return cells


# added function
def get_cells_gt(img, img_dir):
    annot_dir = os.path.join(DATA_DIR, 'test')
    json_path = os.path.join(annot_dir, img.split('.')[0] + '.json')
    gt_bboxes, gt_labels = io_utils.read_cml_json_file(json_path, general_utils.get_class_map('structure'))
    return gt_bboxes


# refactored
def extract_single_table_metrics(img_name, img_dir, iou_threshold=0.7):
    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    cells = detect_page_cells(image)
    cells_gt = get_cells_gt(img_name, img_dir)
    tp, p_len, g_len = iou_utils.calc_prediction_metrics(cells, cells_gt, iou_threshold)

    precision = float(tp / p_len)
    recall = float(tp / g_len)
    print(f'{img_name}: {precision:.2f} {recall:.2f}')

    return tp, p_len, g_len


# added function
def get_ocr_words(img_name, img_size):
    width, height = img_size
    ocr_name = img_name.replace('.png', '.json')
    ocr_path = os.path.join(OCR_DIR_PATH, ocr_name)
    with open(ocr_path) as f:
        ocr = json.load(f)

    def ocr_word_to_box(word):
        bbox = word['Geometry']['BoundingBox']
        xmin = bbox['Left']
        ymin = bbox['Top']
        xmax = xmin + bbox['Width']
        ymax = ymin + bbox['Height']
        return int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

    ocr_word_boxes = [ocr_word_to_box(item) for item in ocr['Blocks'] if item['BlockType'] == 'WORD']
    print(ocr_word_boxes)
    return ocr_word_boxes


def get_ocr_words_gt(gt_bboxes, gt_labels):
    ocr_word_bboxes = []
    for i in range(len(gt_bboxes)):
        if gt_labels[i] == 'word' or gt_labels[i] == 8:
            ocr_word_bboxes.append(gt_bboxes[i])
    return ocr_word_bboxes


# added function
def show_detected_cells(img_name, img_dir):
    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    cells = detect_page_cells(image)
    cells_gt = get_cells_gt(img_name, img_dir)

    image = draw_on_image(image, cells_gt, 'blue', 5)
    image = draw_on_image(image, cells, 'red', 3)
    image.show()


def get_detected_cells(img_name, img_dir):
    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    cells = detect_page_cells(image)
    return cells


# added function
def show_detected_blanks(img_name, img_dir):
    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    cells = detect_page_cells(image)
    ocr_boxes = get_ocr_words(img_name, image.size)
    image = draw_on_image(image, ocr_boxes, 'grey', 1)

    w, h = image.size
    cluster_diff = h * 0.005

    ocr_list = [('word', item[0], item[1], item[2], item[3]) for item in ocr_boxes]
    edited_cells, clust_c, clusts = textract_utils.clusterise_cells(cells, ocr_list, cluster_diff=cluster_diff)
    blank_cells = textract_utils.find_blanks(edited_cells, ocr_list, cluster_diff)

    image = draw_on_image(image, blank_cells, 'blue', 3)
    image.show()


def get_detected_blanks(img_name):
    img_dir = os.path.join(DATA_DIR, 'images')
    annot_dir = os.path.join(DATA_DIR, 'test')

    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    cells = detect_page_cells(image)
    cells_gt, labels_gt = general_utils.get_cells_gt(img_name, annot_dir, with_labels=True)
    ocr_boxes = get_ocr_words_gt(cells_gt, labels_gt)

    w, h = image.size
    cluster_diff = h * 0.005

    ocr_list = [('word', item[0], item[1], item[2], item[3]) for item in ocr_boxes]
    edited_cells, clust_c, clusts = textract_utils.clusterise_cells(cells, ocr_list, cluster_diff=cluster_diff)
    blank_cells = textract_utils.find_blanks(edited_cells, ocr_list, cluster_diff)
    return blank_cells


def ocr_from_gt(gt_bboxes, gt_labels):
    ocr_bboxes = []
    for i in range(len(gt_bboxes)):
        if gt_labels[i] == 'word' or gt_labels[i] == 8:
            ocr_bboxes.append(gt_bboxes[i])
    return ocr_bboxes


def extract_table_structure_with_metrics():
    img_dir = './docs/original_dataset'
    iou_threshold = 0.7
    print(img_dir)
    print(f'IoU: {iou_threshold}')
    images = os.listdir(img_dir)
    all_tp = 0
    all_p_len = 0
    all_g_len = 0
    for img in images:
        if img.endswith('.png'):
            tp, p_len, g_len = extract_single_table_metrics(img, img_dir, iou_threshold)
            all_tp += tp
            all_p_len += p_len
            all_g_len += g_len
    average_precision = all_tp / all_p_len
    average_recall = all_tp / all_g_len
    print(f'total: {average_precision:.2f} {average_recall:.2f}')


def get_predictions():
    img_dir = os.path.join(DATA_DIR, 'images')
    images = os.listdir(img_dir)
    predict_json = {}
    for img_name in images:
        print(img_name)
        p_cells = get_detected_cells(img_name, img_dir)
        p_blanks = get_detected_blanks(img_name)
        predict_json[img_name] = (p_cells, p_blanks)
    pred_path = os.path.join(DATA_DIR, 'tatr_predict')
    os.makedirs(pred_path, exist_ok=True)
    io_utils.save_json(pred_path, 'prediction', predict_json)


def show_prediction(img_name):
    img_dir = os.path.join(DATA_DIR, 'images')
    pred_dir = os.path.join(DATA_DIR, 'tatr_predict')
    with open(os.path.join(pred_dir, 'prediction.json')) as f:
        pred_dict = json.load(f)
    img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    p_cells, p_blanks = pred_dict[img_name]
    img = image_utils.draw_on_image(img, p_cells, 'blue', 2)
    img = image_utils.draw_on_image(img, p_blanks, 'red', 3)
    img.show()


def get_metric(img_name):
    img_dir = os.path.join(DATA_DIR, 'images')
    annot_dir = os.path.join(DATA_DIR, 'test')

    gt_bboxes, gt_labels = general_utils.get_cells_gt(img_name, annot_dir, with_labels=True)
    p_blanks = get_detected_blanks(img_name)
    gt_blanks = []
    for i in range(len(gt_bboxes)):
        if gt_labels[i] == PARAMETER:
            gt_blanks.append(gt_bboxes[i])
    tp, p_len, g_len = iou_utils.calc_prediction_metrics(p_blanks, gt_blanks, thresh=0.7)
    return img_name, tp, p_len, g_len


def get_metrics():
    img_dir = os.path.join(DATA_DIR, 'images')
    images = os.listdir(img_dir)
    all_tp = 0
    all_p_len = 0
    all_g_len = 0
    for img_name in images:
        img_name, tp, p_len, g_len = get_metric(img_name)
        all_tp += tp
        all_p_len += p_len
        all_g_len += g_len
        prec = 0.0 if p_len == 0 else float(tp / p_len)
        rec = 0.0 if g_len == 0 else float(tp / g_len)
        print(f'{img_name}: {prec:.2f} {rec:.2f}')
    average_precision = all_tp / all_p_len
    average_recall = all_tp / all_g_len
    print(f'total: {average_precision:.2f} {average_recall:.2f}')


def get_metric_from_file(img_name, pred_args, with_cells=False, with_blanks=True):
    img_dir = os.path.join(DATA_DIR, 'images')
    annot_dir = os.path.join(DATA_DIR, 'test')

    p_cells, p_blanks = pred_args

    gt_bboxes, gt_labels = general_utils.get_cells_gt(img_name, annot_dir, with_labels=True)
    gt_used = []
    p_used = []
    if with_cells:
        for i in range(len(gt_bboxes)):
            if gt_labels[i] == 5 or gt_labels[i] == 'table spanning cell':
                gt_used.append(gt_bboxes[i])
        p_used.extend(p_cells)
    if with_blanks:
        for i in range(len(gt_bboxes)):
            if gt_labels[i] == 7 or gt_labels[i] == 'blank':
                gt_used.append(gt_bboxes[i])
        p_used.extend(p_blanks)

    tp, p_len, g_len = iou_utils.calc_prediction_metrics(p_used, gt_used, thresh=0.7)
    return img_name, tp, p_len, g_len


def get_metrics_from_file():
    img_dir = os.path.join(DATA_DIR, 'images')
    pred_dir = os.path.join(DATA_DIR, 'tatr_predict')
    images = os.listdir(img_dir)
    with open(os.path.join(pred_dir, 'prediction.json')) as f:
        pred_dict = json.load(f)
    all_tp = 0
    all_p_len = 0
    all_g_len = 0
    for img_name in images:
        img_name, tp, p_len, g_len = get_metric_from_file(img_name, pred_dict[img_name], with_cells=False, with_blanks=True)
        all_tp += tp
        all_p_len += p_len
        all_g_len += g_len
        prec = 0.0 if p_len == 0 else float(tp / p_len)
        rec = 0.0 if g_len == 0 else float(tp / g_len)
        print(f'{img_name}: {prec:.2f} {rec:.2f}')
    average_precision = all_tp / all_p_len
    average_recall = all_tp / all_g_len
    print(f'total: {average_precision:.2f} {average_recall:.2f}')



OCR_DIR_PATH = './docs/document_ocr'
if __name__ == '__main__':
    # show_detected_cells('000001.png', './docs/original_dataset')
    # show_detected_blanks('000001.png', './docs/original_dataset')
    # extract_table_structure_with_metrics()
    # get_predictions()
    show_prediction('D:/Work/table-transformer/resources/structure/val/images/000001_0.png')
    # get_metrics_from_file()
