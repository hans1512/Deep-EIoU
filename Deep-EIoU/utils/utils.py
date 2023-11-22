import numpy as np
from sklearn.cluster import KMeans


def adjust_bbox(image, bbox, height_fraction=(0.17, 0.5), width_fraction=(0.2, 0.2)):
    x1, y1, x2, y2 = bbox

    # Adjusting the bounding box to focus on the upper part
    height = y2 - y1
    width = x2 - x1

    new_y1 = y1 + int(height_fraction[0] * height)
    new_y2 = y1 + int(height_fraction[1] * height)
    new_x1 = x1 + int(width_fraction[0] * width)
    new_x2 = x2 - int(width_fraction[1] * width)

    adjusted_bbox = (new_x1, new_y1, new_x2, new_y2)
    roi = image[adjusted_bbox[1]:adjusted_bbox[3], adjusted_bbox[0]:adjusted_bbox[2]]

    return roi


def perform_kmeans_clustering(colors, initial_centroids, color_multiplier=4, max_iter=10, n_clusters=2):
    """
    Perform k-means clustering on the given color data.
    """
    colors = np.array(colors)

    colors[:, 1:3] *= 4
    if len(initial_centroids) != 0:
        kmeans = KMeans(n_clusters=2, init=initial_centroids, random_state=0, n_init=1, max_iter=max_iter).fit(colors)
    else:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
        initial_centroids = kmeans.cluster_centers_

    # Reassign labels based on the order of centroids
    sorted_idx = np.argsort(initial_centroids[:, 0])  # Assuming the 0th feature is a color channel like Red
    mapping = np.zeros(n_clusters, dtype=int)
    for i, idx in enumerate(sorted_idx):
        mapping[idx] = i

    consistent_labels = mapping[kmeans.labels_]

    return consistent_labels, initial_centroids


def split_ball_players(detections):
    dets = []
    ball = np.array([])
    highest_conf = 0

    for bounding_boxes in detections:
        if bounding_boxes[-1:] != 0:
            dets.append(bounding_boxes)
        if bounding_boxes[5] == 0:
            if bounding_boxes[4] > highest_conf:
                highest_conf = bounding_boxes[4]
                ball = bounding_boxes

    dets = np.array(dets)

    return dets, ball

