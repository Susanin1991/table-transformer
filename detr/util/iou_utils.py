import os

import io_utils
import data_pipeline


def bbox_area(bbox: list):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return width * height


def intersection_area(bbox1: list, bbox2: list):
    x1min, y1min, x1max, y1max = bbox1
    x2min, y2min, x2max, y2max = bbox2
    ximin = max(x1min, x2min)
    yimin = max(y1min, y2min)
    ximax = min(x1max, x2max)
    yimax = min(y1max, y2max)
    if ximin >= ximax or yimin >= yimax:
        return 0
    return bbox_area([ximin, yimin, ximax, yimax])


def union_area(bbox1: list, bbox2: list):
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    area_i = intersection_area(bbox1, bbox2)
    return area1 + area2 - area_i


def intersection_over_union(bbox1, bbox2):
    area_i = intersection_area(bbox1, bbox2)
    area_u = union_area(bbox1, bbox2)
    return float(area_i / area_u)


def is_inside(bbox_out, bbox_in, thresh=0.5):
    in_area = bbox_area(bbox_in)
    inter_area = intersection_area(bbox_out, bbox_in)
    iob = inter_area / in_area
    iou = intersection_over_union(bbox_out, bbox_in)
    if iob >= thresh and iou < 0.5:
        return True
    return False


def calc_prediction_metrics(bboxes_pred: list, bboxes_ground: list, thresh):
    gt = bboxes_ground.copy()
    pred_len = len(bboxes_pred)
    ground_len = len(bboxes_ground)
    true_positives = 0
    for bbox_pred in bboxes_pred:
        max_iou = 0
        max_iou_bbox = None
        for bbox_ground in gt:
            iou = intersection_over_union(bbox_pred, bbox_ground)
            if iou > max_iou:
                max_iou = iou
                max_iou_bbox = bbox_ground
        if max_iou_bbox is not None and max_iou >= thresh:
            gt.remove(max_iou_bbox)
            true_positives += 1
    return true_positives, pred_len, ground_len


def metrics_for_cell(bbox, bboxes_ground, thresh):
    max_iou = 0
    max_iou_bbox = None
    for bbox_ground in bboxes_ground:
        iou = intersection_over_union(bbox, bbox_ground)
        if iou > max_iou:
            max_iou = iou
            max_iou_bbox = bbox_ground
    if max_iou_bbox is not None and max_iou >= thresh:
        return True
    return False


def from_objects_to_bboxes(objects):
    bboxes = []
    for obj in objects:
        bboxes.append(obj['bbox'])
    return bboxes


def from_annotations_to_bboxes(annotations):
    bboxes = []
    for obj in annotations:
        bbox = io_utils.cml_coordinates_to_bbox(obj['coordinates'])
        bboxes.append(bbox)
    return bboxes


def from_cv2_cordinates_to_bboxes(cordinates):
    bboxes = []
    for cord in cordinates:
        xmin, ymin, w, h = cord
        xmax = xmin + w
        ymax = ymin + h
        bbox = [xmin, ymin, xmax, ymax]
        bboxes.append(bbox)
    return bboxes


def from_bboxes_to_cv2_cords(bboxes):
    cords = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        cord = (xmin, ymin, w, h)
        cords.append(cord)
    return cords


def metrics_from_cv2(folder_pred, folder_ground, iou_thresh):
    print(f'threshold: {iou_thresh}')
    p_files = os.listdir(folder_pred)
    all_tp = 0
    all_p_len = 0
    all_g_len = 0
    mpf = []
    for file in p_files:
        if file.endswith('.json'):
            p_bboxes, labels = io_utils.read_cml_json_file(os.path.join(folder_pred, file),
                                                           data_pipeline.get_class_map('structure'))
            try:
                g_bboxes, labels = io_utils.read_cml_json_file(os.path.join(folder_ground, file),
                                                               data_pipeline.get_class_map('structure'))
            except:
                break
            tp, p_len, g_len = calc_prediction_metrics(p_bboxes, g_bboxes, iou_thresh)
            precision = float(tp / p_len)
            recall = float(tp / g_len)
            all_tp += tp
            all_p_len += p_len
            all_g_len += g_len
            mpf.append((file, precision, recall, p_bboxes))
            print(f'{file}: {precision:.2f} {recall:.2f}')
    average_precision = all_tp / all_p_len
    average_recall = all_tp / all_g_len
    mpf.append((average_precision, average_recall))
    print(f'total: {average_precision:.2f} {average_recall:.2f}')
    return mpf


if __name__ == '__main__':
    cv2_folder = '../docs/cv2_prediction'
    ground_folder = '../docs/structure_dataset'
    metrics_from_cv2(cv2_folder, ground_folder, 0.7)
