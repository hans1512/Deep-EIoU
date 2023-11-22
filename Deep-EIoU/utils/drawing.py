from collections import Counter

import cv2
import numpy as np

from utils.Color import Color
from utils.Rectangle import Rect
from utils.utils import adjust_bbox


def get_color(cls_id):
    np.random.seed(cls_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


def filter_out_color_LAB(roi_lab, lower_treshold, upper_treshold):
    # Getting the A channel and applying threshold
    a_channel = roi_lab[:, :, 1]
    # 127, 255
    _, mask = cv2.threshold(a_channel, lower_treshold, upper_treshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Using the mask to exclude green pixels in LAB
    roi_lab_filtered = roi_lab.copy()
    roi_lab_filtered[mask != 255] = 0
    return roi_lab_filtered


def get_mean_color(image, bbox):
    """
    Get the mean color within a specific portion of the bounding box in both RGB and LAB color spaces,
    excluding green pixels.
    """
    roi = adjust_bbox(image, bbox)
    # Converting to LAB color space
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    roi_lab_filtered = filter_out_color_LAB(roi_lab, 127, 255)

    non_zero_subarrays = roi_lab_filtered[roi_lab_filtered.any(axis=-1)]

    # Calculate the mean of each channel
    if non_zero_subarrays.size > 0:
        mean_color_lab = non_zero_subarrays.mean(axis=0)

    color_counter = Counter(map(tuple, non_zero_subarrays))
    most_common_color_lab = np.array(color_counter.most_common(1)[0][0], dtype=np.uint8)

    # most_common_color_bgr = cv2.cvtColor(np.array([[most_common_color_lab]], dtype=np.uint8), cv2.COLOR_Lab2BGR)[0][0]
    # mean_color_bgr = cv2.cvtColor(np.array([[mean_color_lab]], dtype=np.uint8), cv2.COLOR_Lab2BGR)[0][0]

    return most_common_color_lab, mean_color_lab


def draw_ellipse(image, rect, color, thickness=2):
    center = rect.bottom_center
    axes = (int(rect.width), int(0.35 * rect.width))
    cv2.ellipse(
        image,
        center,
        axes,
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image


def plot_tracking_on_frame(image, xyxys, ids, clss, labels, ball):
    thickness = 2
    fontscale = 0.5

    for xyxy, id, cls, label in zip(xyxys, ids, clss, labels):
        # Convert bounding box coordinates to Rect object
        rect = Rect(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
        if cls == 1:
            cls = cls + label

        # Get a unique color for each class
        color = Color(*get_color(cls))

        # Draw an ellipse at the bottom of the bounding box
        draw_ellipse(image, rect, color, thickness)

        # Place ID text just above the ellipse
        cv2.putText(
            image,
            f'{id}',
            (rect.bottom_center[0] - 10, rect.bottom_center[1] - int(0.35 * rect.width) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            color.bgr_tuple,
            thickness
        )
    if len(ball) != 0:
        rect = Rect(int(ball[0]), int(ball[1]), int(ball[2]), int(ball[3]))
        # Get a unique color for each class
        color = Color(*get_color(0))

        # Draw an ellipse at the bottom of the bounding box
        draw_ellipse(image, rect, color, thickness)
        # Place ID text just above the ellipse
        cv2.putText(
            image,
            "-1",
            (rect.bottom_center[0] - 10, rect.bottom_center[1] - int(0.35 * rect.width) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            color.bgr_tuple,
            thickness
        )
    return image
