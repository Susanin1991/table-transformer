import os.path

import cv2
import numpy
import numpy as np
from PIL import Image

import io_utils
import data_pipeline
import image_utils
import pytesseract

import iou_utils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 1)


def thresholding(image, thresh):
    # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)[1]


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def invert(image):
    return cv2.bitwise_not(image)


def find_cords(img, contours, padding=10):
    cordinates = []
    i = 1
    h_im, w_im, c = img.shape
    table_area = (w_im - padding) * (h_im - padding)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # if float(h / h_im) < 0.95 or float(w / w_im) < 0.95:
        # if (float(w / w_im) > 0.03 and float(h / h_im) > 0.03) and (float(h / h_im) < 0.9 or float(w / w_im) < 0.9):
        cell_area = w * h
        if (float(cell_area) > float(table_area / 200)) and (float(cell_area) < float(table_area / 4)) and (float(w / w_im) > 0.03 and float(h / h_im) > 0.03):
            cordinates.append((x, y, w, h))
            i += 1
    return cordinates


def get_cells(folder, img_path, thresh=180, padding=(10, 10, 10, 10), op=False, denoise=False):
    img = os.path.join(folder, img_path)
    im = cv2.imread(img)
    h_im, w_im, c = im.shape
    r_pad, t_pad, l_pad, b_pad = padding
    im = cv2.rectangle(im, (r_pad, t_pad), (w_im - l_pad, h_im - b_pad), (0, 0, 0), 2)
    res = get_grayscale(im)
    if op:
        res = opening(res)
    if denoise:
        res = remove_noise(res)
    thr = thresholding(res, thresh)
    dil = dilate(thr)
    contours = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cordinates = find_cords(im, contours)
    return cordinates
    # plot = plt.imshow(dil)
    # plt.show()


def get_cells_v2(img, thresh=180, padding=(10, 10, 10, 10), op=False, denoise=False):
    image = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    h_im, w_im, c = image.shape
    r_pad, t_pad, l_pad, b_pad = padding
    image = cv2.rectangle(image, (r_pad, t_pad), (w_im - l_pad, h_im - b_pad), (0, 0, 0), 2)
    res = get_grayscale(image)
    if op:
        res = opening(res)
    if denoise:
        res = remove_noise(res)
    thr = thresholding(res, thresh)
    dil = dilate(thr)
    contours = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cordinates = find_cords(image, contours)
    return cordinates
    # plot = plt.imshow(dil)
    # plt.show()


def get_text_bboxes(folder, img_path, conf_thresh=60):
    img = cv2.imread(os.path.join(folder, img_path))
    h_im, w_im, c = img.shape
    cordinates = []
    output = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 11')
    n_boxes = len(output['text'])
    # print(n_boxes)
    for i in range(n_boxes):
        if int(output['conf'][i]) > conf_thresh:
            (x, y, w, h) = (output['left'][i], output['top'][i], output['width'][i], output['height'][i])
            cordinates.append((x, y, w, h))
            # if int(output['conf'][i]) < 60:
            #     print(output['text'][i], output['conf'][i])
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # plot = plt.imshow(img)
    # plt.show()
    return cordinates


def exclude_text_bboxes(cords_cv2, cords_tes):
    new_cords = []
    for cv2_bbox in cords_cv2:
        xmin, ymin, w, h = cv2_bbox
        xmax = xmin + w
        ymax = ymin + h
        bbox = [xmin, ymin, xmax, ymax]
        tes_bboxes = iou_utils.from_cv2_cordinates_to_bboxes(cords_tes)
        found = iou_utils.metrics_for_cell(bbox, tes_bboxes, 0.7)
        if not found:
            new_cords.append(cv2_bbox)
    return new_cords


def exclude_bboxes(cords):
    bboxes = iou_utils.from_cv2_cordinates_to_bboxes(cords)
    new_bboxes = bboxes.copy()
    new_cords = []
    for bbox in bboxes:
        removal_bboxes =[]
        for b in new_bboxes:
            bbox_area = iou_utils.bbox_area(b)
            inter_area = iou_utils.intersection_area(bbox, b)
            iob = inter_area / bbox_area
            iou = iou_utils.intersection_over_union(bbox, b)
            if iob >= 0.5 and iou < 0.5:
                removal_bboxes.append(b)
        for r in removal_bboxes:
            new_bboxes.remove(r)
    for bbox in new_bboxes:
        new_cords.append((bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return new_cords


def inside_error(bbox1: list, bbox2: list, error: int):
    xmin_dif = abs(bbox1[0] - bbox2[0])
    ymin_dif = abs(bbox1[1] - bbox2[1])
    xmax_dif = abs(bbox1[2] - bbox2[2])
    ymax_dif = abs(bbox1[3] - bbox2[3])
    if (xmin_dif <= error and xmax_dif <= error) or (ymin_dif <= error and ymax_dif <= error):
        return True
    return False


def add_bboxes(cell_bbox: list, bboxes: list[list], thresh=0.5, error=5):
    for bbox in bboxes:
        if iou_utils.intersection_over_union(cell_bbox, bbox) == 1:
            pass
        if iou_utils.intersection_over_union(cell_bbox, bbox) >= thresh and inside_error(cell_bbox, bbox, error):
            xmin = min(cell_bbox[0], bbox[0])
            ymin = min(cell_bbox[1], bbox[1])
            xmax = max(cell_bbox[2], bbox[2])
            ymax = max(cell_bbox[3], bbox[3])
            cell_bbox = [xmin, ymin, xmax, ymax]
    return cell_bbox


def merge_cells(cords):
    edited_bboxes = []
    bboxes = iou_utils.from_cv2_cordinates_to_bboxes(cords)
    for bbox in bboxes:
        bbox = add_bboxes(bbox, bboxes, thresh=0.1)
        edited_bboxes.append(bbox)
    cords = iou_utils.from_bboxes_to_cv2_cords(edited_bboxes)
    cords = exclude_bboxes(cords)
    return cords


def get_text_v2():
    ocr_json = pytesseract.image_to_data(img, output_type='dict', config='--psm 11')
    words = []
    for i in range(len(ocr_json['width'])):
        words.append((ocr_json['left'][i],
                      ocr_json['top'][i],
                      ocr_json['left'][i] + ocr_json['width'][i],
                      ocr_json['top'][i] + ocr_json['height'][i],
                      ocr_json['text'][i]))


def save_cv2_as_json(folder, img_path, coordinates):
    json_obj = io_utils.cv2cords_to_cml(img_path, coordinates)
    io_utils.save_json(folder, img_path, [json_obj])


def compare_with_gt_cv2f(gt_div, img_name, cords):
    image = Image.open(os.path.join(gt_div, img_name))
    bboxes, labels = image_utils.read_createml_json_file(os.path.join(gt_div, img_name), data_pipeline.get_class_map('structure'))
    image_utils.draw_on_image(image, bboxes, 'blue', 6)
    cv_bboxes = iou_utils.from_cv2_cordinates_to_bboxes(cords)
    image_utils.draw_on_image(image, cv_bboxes, 'red', 2)
    image_utils.visualise_image(image)


def compare_with_gt_bbf(gt_div, img_name, p_bboxes):
    img_path = os.path.join(gt_div, img_name)
    image = Image.open(img_path)
    g_bboxes, labels = image_utils.read_createml_json_file(os.path.join(gt_div, img_name.split('.')[0] + '.json'),
                                                           data_pipeline.get_class_map('structure'))
    image_utils.draw_on_image(image, g_bboxes, 'blue', 6)
    image_utils.draw_on_image(image, p_bboxes, 'red', 2)
    image_utils.visualise_image(image)


if __name__ == '__main__':
    folder = '../docs/structure_dataset'
    img = '000007_0.png'
    cords = get_cells(folder, img, thresh=180, padding=(25, 10, 20, 10))
    image_utils.draw_image_bboxes_wh(os.path.join(folder, img), cords)

    new_cords = exclude_bboxes(cords)
    image_utils.draw_image_bboxes_wh(os.path.join(folder, img), new_cords)

    # merged_cords = merge_cells(new_cords)
    # image_utils.draw_image_bboxes_wh(os.path.join(folder, img), merged_cords)

    # txt_cords = get_text_bboxes(folder, img)
    # image_utils.draw_image_bboxes_wh(os.path.join(folder, img), txt_cords)

    # final = exclude_text_bboxes(new_cords, txt_cords)
    # image_utils.draw_image_bboxes_wh(os.path.join(folder, img), final)
