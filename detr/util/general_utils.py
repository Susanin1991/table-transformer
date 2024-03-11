import multiprocessing
import os

from tqdm import tqdm

import io_utils


def run_multiprocessing(task, items):
    print('run multiprocessing')
    print(items)

    num_procs = os.cpu_count()  # number of CPU cores
    print(num_procs)

    pool = multiprocessing.Pool(num_procs)
    results = []
    for result in tqdm(pool.imap_unordered(task, items), total=len(items)):
        results.append(result)

    return results


def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6,
            'blank': 7,
            'word': 8
        }
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map


def get_cells_gt(img_name, annot_path, with_labels=False):
    json_path = os.path.join(annot_path, img_name.split('.')[0] + '.json')
    gt_bboxes, gt_labels = io_utils.read_cml_json_file(json_path, get_class_map('structure'))
    if with_labels:
        return gt_bboxes, gt_labels
    return gt_bboxes
