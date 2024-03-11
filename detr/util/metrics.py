import os

import general_utils
import io_utils

DATA_DIR = './dataset'
PARAMETER = 7


# def metrics_for_table(predicted_bboxes, predicted_labels):
#     gt_bboxes, gt_labels = io_utils.read_cml_json_file(os.path.join(test_dir, file),
#                                                        general_utils.get_class_map('structure'))
#     used_gt_bboxes = []
#     used_p_bboxes = []
#     for i in range(len(gt_bboxes)):
#         if gt_labels[i] == PARAMETER:
#             used_gt_bboxes.append(gt_bboxes[i])
#     for i in range(len(edited_cells)):
#         if labels[i] == PARAMETER:
#             used_p_bboxes.append(edited_cells[i])
#     tp, p_len, g_len = metrics_from_cv2(used_p_bboxes, used_gt_bboxes, iou_thresh=0.7)
#     return file, tp, p_len, g_len, edited_cells, labels, gt_bboxes, gt_labels

def get_metrics(data_dir):
    test_dir = os.path.join(data_dir, 'test')
    files = os.listdir(test_dir)
    all_tp = 0
    all_p_len = 0
    all_g_len = 0
    mpf = []
    results = general_utils.run_multiprocessing(metrics_for_table, files[:100])
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