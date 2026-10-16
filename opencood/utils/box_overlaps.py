import numpy as np


def bbox_overlaps(boxes, query_boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
    for k in range(query_boxes.shape[0]):
        query = query_boxes[k]
        query_area = (query[2] - query[0] + 1) * (query[3] - query[1] + 1)
        for n in range(boxes.shape[0]):
            box = boxes[n]
            iw = min(box[2], query[2]) - max(box[0], query[0]) + 1
            if iw <= 0:
                continue
            ih = min(box[3], query[3]) - max(box[1], query[1]) + 1
            if ih <= 0:
                continue
            box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            union = box_area + query_area - iw * ih
            if union > 0:
                overlaps[n, k] = iw * ih / union
    return overlaps


def bbox_intersections(boxes, query_boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    intersections = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
    for k in range(query_boxes.shape[0]):
        query = query_boxes[k]
        query_area = (query[2] - query[0] + 1) * (query[3] - query[1] + 1)
        if query_area <= 0:
            continue
        for n in range(boxes.shape[0]):
            box = boxes[n]
            iw = min(box[2], query[2]) - max(box[0], query[0]) + 1
            if iw <= 0:
                continue
            ih = min(box[3], query[3]) - max(box[1], query[1]) + 1
            if ih > 0:
                intersections[n, k] = iw * ih / query_area
    return intersections
