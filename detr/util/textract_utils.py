import json
import os
import sys

import boto3
from PIL import Image

import general_utils
import io_utils
import data_pipeline
import image_utils
import iou_utils


def words_n_bboxes_to_inner_str(words: list, bboxes: list) -> list:
    inner_str = []
    for i in range(len(bboxes)):
        word = words[i]
        xmin, ymin, xmax, ymax = bboxes[i]
        inner_str.append((word, xmin, ymin, xmax, ymax))
    return inner_str


def get_text_textract(folder, img):
    with open(os.path.join(folder, img), 'rb') as img_file:
        f = img_file.read()
        return recognize_text(f)


def recognize_text(data: bytes) -> dict:
    tr_client = boto3.client('textract')
    response = tr_client.detect_document_text(Document={'Bytes': data})
    return response


def ocr_to_words(ocr: dict, width: int, height: int):
    words = []
    bboxes = []
    for block in ocr['Blocks']:
        if block['BlockType'] == 'WORD':
            words.append(block['Text'])
            geom = block['Geometry']['BoundingBox']
            xmin = geom['Left'] * width
            ymin = geom['Top'] * height
            w = geom['Width'] * width
            h = geom['Height'] * height
            xmax = xmin + w
            ymax = ymin + h
            bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes, words


def load_ocr(folder, name):
    file_path = os.path.join(folder, name)
    with open(file_path) as f:
        ocr = json.load(f)
    return ocr


def find_cell_words(cell_bbox: list, ocr_list: list, expand=False):
    cell_words = []
    for item in ocr_list:
        word, xmin, ymin, xmax, ymax = item
        bbox = [xmin, ymin, xmax, ymax]
        if expand:
            cell_bbox = expand_cell_by_word(cell_bbox, bbox, 0.5)
        if iou_utils.is_inside(cell_bbox, bbox, thresh=0.95):
            cell_words.append(item)
    return cell_bbox, cell_words


def cluster_ymin(cluster):
    ymin_min = cluster[0][2]
    ymin_max = cluster[0][2]
    for bbox in cluster:
        if bbox[2] > ymin_max:
            ymin_max = bbox[2]
        if bbox[2] < ymin_min:
            ymin_min = bbox[2]
    return ymin_min, ymin_max


def divide_to_clusters(cell_words: list, cluster_dif):
    clusters = [[cell_words.pop(0)]]
    for word in cell_words:
        min_diff = sys.maxsize
        min_cluster = None
        for cluster in clusters:
            for bbox in cluster:
                diff = abs(word[4] - bbox[4])
                if diff < min_diff:
                    min_diff = diff
                    min_cluster = cluster
        if min_diff < cluster_dif and min_cluster is not None:
            min_cluster.append(word)
        else:
            clusters.append([word])
    return clusters


def clusterise_in_cell(cell_bbox: list, ocr_list: list, cluster_dif):
    cell_bbox, cell_words = find_cell_words(cell_bbox, ocr_list)
    if len(cell_words) == 0:
        return cell_bbox, 0, []
    clusters = divide_to_clusters(cell_words, cluster_dif)
    return cell_bbox, len(clusters), clusters


def clusterise_cells(doc_bboxes, ocr_list, cluster_diff=25):
    page_clusters = []
    page_clust_c = []
    edited_cells = []
    for bbox in doc_bboxes:
        bbox, clus_c, cell_clusters = clusterise_in_cell(bbox, ocr_list, cluster_diff)
        page_clusters.append(cell_clusters)
        page_clust_c.append(clus_c)
        edited_cells.append(bbox)
    return edited_cells, page_clust_c, page_clusters


def cluster_bbox(cluster):
    xmin = 1000000
    ymin = 1000000
    xmax = -1
    ymax = -1
    for bbox in cluster:
        if bbox[1] < xmin:
            xmin = bbox[1]
        if bbox[2] < ymin:
            ymin = bbox[2]
        if bbox[3] > xmax:
            xmax = bbox[3]
        if bbox[4] > ymax:
            ymax = bbox[4]
    return [xmin, ymin, xmax, ymax]


def get_cluster_bboxes(page_clusters: list):
    cluster_bboxes = []
    for page_cluster in page_clusters:
        for cell_cluster in page_cluster:
            bbox = cluster_bbox(cell_cluster)
            cluster_bboxes.append(bbox)
    return cluster_bboxes


def max_bbox(bboxes: list):
    max_b = None
    max_area = 0
    for bbox in bboxes:
        area = iou_utils.bbox_area(bbox)
        if area > max_area:
            max_area = area
            max_b = bbox
    return max_b, max_area


def blank_bbox(cell_bbox, cluster, thresh=0.5):
    c_bbox = cluster_bbox(cluster)
    blank_bbox_top = [cell_bbox[0], c_bbox[3], cell_bbox[2], cell_bbox[3]]
    blank_area_top = iou_utils.bbox_area(blank_bbox_top)
    blank_bbox_bottom = [cell_bbox[0], cell_bbox[1], cell_bbox[2], c_bbox[1]]
    blank_bbox_left = [c_bbox[2], cell_bbox[1], cell_bbox[2], cell_bbox[3]]
    cell_area = iou_utils.bbox_area(cell_bbox)
    blank_bbox, blank_area = max_bbox([blank_bbox_top, blank_bbox_bottom, blank_bbox_left])
    if blank_area_top >= blank_area * 0.7:
        blank_area = blank_area_top
        blank_bbox = blank_bbox_top
    if blank_area < thresh * cell_area:
        return False, []
    return True, blank_bbox


def blanks_in_cell(cell_bbox: list, ocr_list: list, cluster_dif):
    cell_bbox, c, clusters = clusterise_in_cell(cell_bbox, ocr_list, cluster_dif)
    if c == 0:
        return True, cell_bbox
    if c == 1:
        return blank_bbox(cell_bbox, clusters[0], thresh=0.4)
    return False, []


def find_blanks(doc_bboxes, ocr_list, cluster_dif):
    blank_cells = []
    for doc_bbox in doc_bboxes:
        is_blank, bbox = blanks_in_cell(doc_bbox, ocr_list, cluster_dif)
        if is_blank:
            blank_cells.append(bbox)
    return blank_cells


def expand_cell_by_word(cell_bbox, word_bbox, thresh=0.5):
    if iou_utils.is_inside(cell_bbox, word_bbox, thresh):
        xmin = min(cell_bbox[0], word_bbox[0])
        ymin = min(cell_bbox[1], word_bbox[1])
        xmax = max(cell_bbox[2], word_bbox[2])
        ymax = max(cell_bbox[3], word_bbox[3])
        return [xmin, ymin, xmax, ymax]
    return cell_bbox


if __name__ == '__main__':
    file_name = '000003'
    img = Image.open(os.path.join('../docs/original_dataset', file_name + '.png'))
    w, h = img.width, img.height
    cluster_diff = h * 0.006

    ocr = load_ocr('../docs/document_ocr', file_name + '.json')
    ocr_bboxes, ocr_words = ocr_to_words(ocr, w, h)

    ocr_list = words_n_bboxes_to_inner_str(ocr_words, ocr_bboxes)
    bboxes, labels = io_utils.read_cml_json_file(f'../docs/original_dataset/{file_name}.json',
                                                 general_utils.get_class_map('structure'))

    edited_cells, clust_c, clusts = clusterise_cells(bboxes, ocr_list, cluster_diff=cluster_diff)
    clust_bboxes = get_cluster_bboxes(clusts)
    blank_cells = find_blanks(bboxes, ocr_list, cluster_diff)

    image_utils.draw_on_image(img, edited_cells, 'green', 2)
    # image_utils.draw_on_image(img, ocr_bboxes, 'blue', 1)
    image_utils.draw_on_image(img, clust_bboxes, 'purple', 2)
    image_utils.draw_on_image(img, blank_cells, 'red', 3)
    image_utils.visualise_image(img)
