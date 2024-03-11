import os

from PIL import Image
from tqdm import tqdm

import general_utils
import io_utils
import cv2util
import data_pipeline
import image_utils
import iou_utils
import textract_utils


DATA_DIR = './dataset'
PARAMETER = 7


def crop_to_single_tables(img_dir, annot_dir, img_name, padding=10, thresh=180, det_padding=(23, 10, 22, 10)):
    table_bboxes = []
    cropped_dicts = []
    class_map = general_utils.get_class_map('structure')
    full_img_path = os.path.join(img_dir, img_name)
    img = Image.open(full_img_path)
    file_name = img_name.split('.')[0]
    full_annot_path = os.path.join(annot_dir, f'{file_name}.json')
    bboxes, labels = io_utils.read_cml_json_file(full_annot_path, class_map)
    for i in range(len(bboxes)):
        if labels[i] == 'table' or labels[i] == 0:
            bbox = [bboxes[i][0] - padding, bboxes[i][1] - padding, bboxes[i][2] + padding, bboxes[i][3] + padding]
            cropped_img = img.crop(bbox)
            json_dict = create_structure_with_cv2_v2(cropped_img, img_name, thresh=thresh, padding=det_padding)
            table_bboxes.append(bboxes[i])
            cropped_dicts.append(json_dict)
    return table_bboxes, cropped_dicts


def merge_tables(img, img_name: str, table_bboxes: list, cropped_dicts: list):
    class_map = general_utils.get_class_map('structure')
    json_dict = io_utils.write_cml_json(img_name, [], [], class_map)
    for i in range(len(table_bboxes)):
        crop_dict = cropped_dicts[i]
        json_dict = data_pipeline.move_crop_cords(img, table_bboxes[i], json_dict, crop_dict)
    return json_dict


def create_structure_with_cv2(folder, img_name, denoise=False, thresh=180, padding=(23, 10, 22, 10)):
    cords = cv2util.get_cells(folder, img_name, denoise=denoise, thresh=thresh, padding=padding)
    n_cords = cv2util.exclude_bboxes(cords)
    final = n_cords
    json_dict = io_utils.cv2cords_to_cml(img_name, final)
    return json_dict


def create_structure_with_cv2_v2(img, img_name, denoise=False, thresh=180, padding=(23, 10, 22, 10)):
    cords = cv2util.get_cells_v2(img, denoise=denoise, thresh=thresh, padding=padding)
    n_cords = cv2util.exclude_bboxes(cords)
    final = n_cords
    json_dict = io_utils.cv2cords_to_cml(img_name, final)
    return json_dict


def create_doc_structure_with_cv2(img_name, img_dir, annot_dir, thresh=180, padding=(23, 10, 22, 10)):
    img = Image.open(os.path.join(img_dir, img_name))
    table_bboxes, cropped_dicts = crop_to_single_tables(img_dir, annot_dir, img_name, thresh=thresh, det_padding=padding)
    json_dict = merge_tables(img, img_name, table_bboxes, cropped_dicts)
    file = img_name.split('.')[0]
    bboxes, labels = io_utils.read_cml_json_file(os.path.join(annot_dir, f'{file}.json'),
                                                 general_utils.get_class_map('structure'))
    ocr_bboxes = []
    ocr_labels = []
    for i in range(len(bboxes)):
        if labels[i] == 'word' or labels[i] == 8:
            ocr_bboxes.append(bboxes[i])
            ocr_labels.append(labels[i])
    edited_cells, labels, blank_cells, clusts = get_blanks_v2(img, json_dict, ocr_bboxes)
    blank_labels = [7 for i in blank_cells]

    edited_cells.extend(blank_cells)
    edited_cells.extend(ocr_bboxes)
    labels.extend(blank_labels)
    labels.extend(ocr_labels)
    return edited_cells, labels


def get_blanks(file_name, dir_img, dir_not, dir_ocr=''):
    img = Image.open(os.path.join(dir_img, f'{file_name}.png'))
    w, h = img.width, img.height
    cluster_diff = h * 0.005

    if dir_ocr == '':
        ocr = textract_utils.get_text_textract(dir_img, f'{file_name}.png')
    else:
        ocr = textract_utils.load_ocr(dir_ocr, f'{file_name}.json')
    ocr_bboxes, ocr_words = textract_utils.ocr_to_words(ocr, w, h)

    ocr_list = textract_utils.words_n_bboxes_to_inner_str(ocr_words, ocr_bboxes)
    bboxes, labels = io_utils.read_cml_json_file(os.path.join(dir_not, f'{file_name}.json'),
                                                 general_utils.get_class_map('structure'))

    edited_cells, clust_c, clusts = textract_utils.clusterise_cells(bboxes, ocr_list, cluster_diff=cluster_diff)
    clust_bboxes = textract_utils.get_cluster_bboxes(clusts)
    blank_cells = textract_utils.find_blanks(edited_cells, ocr_list, cluster_diff)

    image_utils.draw_on_image(img, edited_cells, 'green', 2)
    # image_utils.draw_on_image(img, ocr_bboxes, 'blue', 1)
    image_utils.draw_on_image(img, clust_bboxes, 'purple', 2)
    image_utils.draw_on_image(img, blank_cells, 'red', 3)
    image_utils.visualise_image(img)

    return edited_cells, labels, blank_cells, clusts


def get_blanks_v2(img, json_dict, ocr_bboxes):
    w, h = img.width, img.height
    cluster_diff = h * 0.005

    ocr_words = ['word' for i in ocr_bboxes]
    ocr_list = textract_utils.words_n_bboxes_to_inner_str(ocr_words, ocr_bboxes)

    bboxes, labels = io_utils.read_cml_json_dict(json_dict, general_utils.get_class_map('structure'))

    edited_cells, clust_c, clusts = textract_utils.clusterise_cells(bboxes, ocr_list, cluster_diff=cluster_diff)
    blank_cells = textract_utils.find_blanks(edited_cells, ocr_list, cluster_diff)
    # clust_bboxes = textract_utils.get_cluster_bboxes(clusts)

    # image_utils.draw_on_image(img, edited_cells, 'green', 2)
    # image_utils.draw_on_image(img, ocr_bboxes, 'blue', 1)
    # image_utils.draw_on_image(img, clust_bboxes, 'purple', 2)
    # image_utils.draw_on_image(img, blank_cells, 'red', 3)
    # image_utils.visualise_image(img)

    return edited_cells, labels, blank_cells, clusts


def metrics_from_cv2(pred_bboxes, gt_bboxes, iou_thresh):
    tp, p_len, g_len = iou_utils.calc_prediction_metrics(pred_bboxes, gt_bboxes, iou_thresh)
    return tp, p_len, g_len


def get_one_metric(file):
    img_dir = os.path.join(DATA_DIR, 'images')
    test_dir = os.path.join(DATA_DIR, 'test')
    file_name = file.split('.')[0]
    gt_bboxes, gt_labels = io_utils.read_cml_json_file(os.path.join(test_dir, file),
                                                       general_utils.get_class_map('structure'))
    edited_cells, labels = create_doc_structure_with_cv2(f'{file_name}.png', img_dir, test_dir, thresh=180)
    used_gt_bboxes = []
    used_p_bboxes = []
    for i in range(len(gt_bboxes)):
        if gt_labels[i] == PARAMETER:
            used_gt_bboxes.append(gt_bboxes[i])
    for i in range(len(edited_cells)):
        if labels[i] == PARAMETER:
            used_p_bboxes.append(edited_cells[i])
    tp, p_len, g_len = metrics_from_cv2(used_p_bboxes, used_gt_bboxes, iou_thresh=0.7)
    return file, tp, p_len, g_len, edited_cells, labels, gt_bboxes, gt_labels


def get_metrics(data_dir):
    test_dir = os.path.join(data_dir, 'test')
    files = os.listdir(test_dir)
    all_tp = 0
    all_p_len = 0
    all_g_len = 0
    mpf = []
    results = general_utils.run_multiprocessing(get_one_metric, files[:100])
    # results = [get_one_metric(files[0])]
    for result in results:
        file, tp, p_len, g_len, p_bboxes, p_labels, gt_bboxes, gt_labels = result
        all_tp += tp
        all_p_len += p_len
        all_g_len += g_len
        prec = 0.0 if p_len == 0 else float(tp / p_len)
        rec = 0.0 if g_len == 0 else float(tp / g_len)
        mpf.append((file, prec, rec, p_bboxes, p_labels, gt_bboxes, gt_labels))
    average_precision = all_tp / all_p_len
    average_recall = all_tp / all_g_len
    mpf.append((average_precision, average_recall))
    print(f'total: {average_precision:.2f} {average_recall:.2f}')
    return mpf


def print_worst_1(mpf):
    min_rec = 1
    min_met = ()
    for i in range(len(mpf) - 1):
        file, prec, rec, p_bboxes, gt_bboxes = mpf[i]
        if rec < min_rec:
            min_rec = rec
            min_met = mpf[i]
    img = Image.open(os.path.join(DATA_DIR, 'images', min_met[0].split('.')[0] + '.png'))
    image_utils.draw_on_image(img, min_met[4], 'blue', 4)
    image_utils.draw_on_image(img, min_met[3], 'red', 2)
    image_utils.visualise_image(img)
    print(min_met[0])


def print_worst(mpf):
    j = 0
    for i in range(len(mpf) - 1):
        file, prec, rec, p_bboxes, p_labels, gt_bboxes, gt_labels = mpf[i]
        p_cells = []
        p_blanks = []
        gt_cells = []
        gt_blanks = []

        for k in range(len(p_bboxes)):
            if p_labels[k] == 7 or p_labels[k] == 'blank':
                p_blanks.append(p_bboxes[k])
            if p_labels[k] == 5 or p_labels[k] == 'table spanning cell':
                p_cells.append(p_bboxes[k])
        for k in range(len(gt_bboxes)):
            if gt_labels[k] == 7 or gt_labels[k] == 'blank':
                gt_blanks.append(gt_bboxes[k])
            if gt_labels[k] == 5 or gt_labels[k] == 'table spanning cell':
                gt_cells.append(gt_bboxes[k])

        if rec == 0 and j <= 2:
            img = Image.open(os.path.join(DATA_DIR, 'images', file.split('.')[0] + '.png'))
            img1 = img.copy()
            image_utils.draw_on_image(img, gt_cells, 'blue', 4)
            image_utils.draw_on_image(img, p_cells, 'red', 2)
            image_utils.visualise_image(img)

            image_utils.draw_on_image(img1, gt_blanks, 'blue', 4)
            image_utils.draw_on_image(img1, p_blanks, 'red', 2)
            image_utils.visualise_image(img1)
            print(f'{file}: {prec:.2f}, {rec:.2f}')
            j += 1


def print_bad(mpf):
    j = 0
    for i in range(len(mpf) - 1):
        file, prec, rec, p_bboxes, p_labels, gt_bboxes, gt_labels = mpf[i]
        p_cells = []
        p_blanks = []
        gt_cells = []
        gt_blanks = []

        for k in range(len(p_bboxes)):
            if p_labels[k] == 7 or p_labels[k] == 'blank':
                p_blanks.append(p_bboxes[k])
            if p_labels[k] == 5 or p_labels[k] == 'table spanning cell':
                p_cells.append(p_bboxes[k])
        for k in range(len(gt_bboxes)):
            if gt_labels[k] == 7 or gt_labels[k] == 'blank':
                gt_blanks.append(gt_bboxes[k])
            if gt_labels[k] == 5 or gt_labels[k] == 'table spanning cell':
                gt_cells.append(gt_bboxes[k])

        if 0 < rec < 0.7 and j <= 2:
            img = Image.open(os.path.join(DATA_DIR, 'images', file.split('.')[0] + '.png'))
            img1 = img.copy()
            image_utils.draw_on_image(img, gt_cells, 'green', 4)
            image_utils.draw_on_image(img, p_cells, 'purple', 2)
            image_utils.visualise_image(img)

            image_utils.draw_on_image(img1, gt_blanks, 'blue', 4)
            image_utils.draw_on_image(img1, p_blanks, 'red', 2)
            image_utils.visualise_image(img1)
            print(f'{file}: {prec:.2f}, {rec:.2f}')
            j += 1


def print_best(mpf):
    max_rec = 0
    max_met = ()
    for i in range(len(mpf) - 1):
        file, prec, rec, p_bboxes, gt_bboxes = mpf[i]
        if rec > max_rec:
            max_rec = rec
            max_met = mpf[i]
    img = Image.open(os.path.join(DATA_DIR, 'images', max_met[0].split('.')[0] + '.png'))
    image_utils.draw_on_image(img, max_met[4], 'blue', 4)
    image_utils.draw_on_image(img, max_met[3], 'red', 2)
    image_utils.visualise_image(img)
    print(max_met[0])


def print_good(mpf):
    j = 0
    for i in range(len(mpf) - 1):
        file, prec, recall, p_bboxes, p_labels, gt_bboxes, gt_labels = mpf[i]
        p_cells = []
        p_blanks = []
        gt_cells = []
        gt_blanks = []

        for k in range(len(p_bboxes)):
            if p_labels[k] == 7 or p_labels[k] == 'blank':
                p_blanks.append(p_bboxes[k])
            if p_labels[k] == 5 or p_labels[k] == 'table spanning cell':
                p_cells.append(p_bboxes[k])
        for k in range(len(gt_bboxes)):
            if gt_labels[k] == 7 or gt_labels[k] == 'blank':
                gt_blanks.append(gt_bboxes[k])
            if gt_labels[k] == 5 or gt_labels[k] == 'table spanning cell':
                gt_cells.append(gt_bboxes[k])

        if recall > 0.8 and prec > 0.8 and j <= 1:
            img = Image.open(os.path.join(DATA_DIR, 'images', file.split('.')[0] + '.png'))
            img1 = img.copy()
            image_utils.draw_on_image(img, gt_cells, 'blue', 5)
            image_utils.draw_on_image(img, p_cells, 'red', 3)
            image_utils.visualise_image(img)

            image_utils.draw_on_image(img1, gt_blanks, 'blue', 5)
            image_utils.draw_on_image(img1, p_blanks, 'red', 3)
            image_utils.visualise_image(img1)
            print(f'{file}: {prec:.2f}, {recall:.2f}')
            j += 1


if __name__ == '__main__':
    # create_doc_structure_with_cv2('000001.png', './docs/original_dataset', './docs/full_ground_truth')
    # get_blanks('000002', './docs/original_dataset', './docs/cv2_doc_prediction', './docs/document_ocr')
    met_per_file = get_metrics('./dataset')
    # print_worst(met_per_file)
    # print_bad(met_per_file)
    print_good(met_per_file)
