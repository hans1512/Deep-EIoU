import os
import os.path as osp
import time
import cv2
import torch
import sys

from ultralytics import YOLO

from utils.drawing import get_mean_color, plot_tracking_on_frame
from utils.utils import perform_kmeans_clustering, split_ball_players

sys.path.append('.')
from utils.config import load_config

from loguru import logger
from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor
from tracker.tracking_utils.timer import Timer

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def inference(model, img, timer):
    img2 = img
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = osp.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    with torch.no_grad():
        timer.tic()
        outputs2 = model.predict(img2, conf=0.6, verbose=False)
    return outputs2, img_info


def imageflow_demo(model, extractor, vis_folder, current_time, config):
    cap = cv2.VideoCapture(config.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, config.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = Deep_EIoU(config, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    initial_centroids = []
    while True:
        if frame_id % 30 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = inference(model, frame, timer)
            if outputs is not None:
                detections = outputs[0].boxes.data.cpu().numpy()

                # Split the ball and player detections
                detections, ball = split_ball_players(detections)
                classes = detections[:, 5].astype('int')

                cropped_imgs = [frame[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))] for
                                x1, y1, x2, y2, _, _ in detections]
                embeddings = extractor(cropped_imgs)
                embeddings = embeddings.cpu().detach().numpy()
                online_targets = tracker.update(detections, embeddings)
                bounding_boxes = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    track_id = t.track_id
                    if tlwh[2] * tlwh[3] > config.min_box_area:
                        bounding_boxes.append(
                            [tlwh[0].astype('int'), tlwh[1].astype('int'), (tlwh[0] + tlwh[2]).astype('int'),
                             (tlwh[1] + tlwh[3]).astype('int')])
                        online_scores.append(t.score)
                        online_ids.append(track_id)
                        results.append(
                            f"{frame_id},{track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()

                player_mean_colors = [get_mean_color(img_info['raw_img'], bounding_box)[1] for bounding_box, cls in
                                      zip(bounding_boxes, classes) if cls == 1]

                if len(initial_centroids) == 0:
                    labels, initial_centroids = perform_kmeans_clustering(player_mean_colors, initial_centroids)
                else:
                    labels, _ = perform_kmeans_clustering(player_mean_colors, initial_centroids)

                online_im = plot_tracking_on_frame(
                    img_info['raw_img'], bounding_boxes, online_ids, classes, labels, online_scores, ball
                )

            else:
                timer.toc()
                online_im = img_info['raw_img']
            if config.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if config.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main():
    config = load_config('tracking_config.json')
    model = YOLO(config.detection_model)

    vis_folder = "tracker_testing"
    os.makedirs(vis_folder, exist_ok=True)

    current_time = time.localtime()

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='checkpoints/sports_model.pth.tar-60',
        device='cuda'
    )

    imageflow_demo(model, extractor, vis_folder, current_time, config)


main()
