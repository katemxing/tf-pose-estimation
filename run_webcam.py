import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger("TfPoseEstimator-WebCam")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tf-pose-estimation realtime webcam")
    parser.add_argument("--camera", type=int, default=0)

    parser.add_argument(
        "--resize",
        type=str,
        default="0x0",
        help="if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ",
    )
    parser.add_argument(
        "--resize-out-ratio",
        type=float,
        default=4.0,
        help="if provided, resize heatmaps before they are post-processed. default=1.0",
    )

    parser.add_argument(
        "--model", type=str, default="mobilenet_thin", help="cmu / mobilenet_thin"
    )

    parser.add_argument(
        "--output_json", type=str, default="/tmp/", help="writing output json dir"
    )

    parser.add_argument(
        "--show-process",
        type=bool,
        default=False,
        help="for debug purpose, if enabled, speed for inference is dropped.",
    )
    args = parser.parse_args()

    logger.debug("initialization %s : %s" % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug("cam read+")
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info("cam image=%dx%d" % (image.shape[1], image.shape[0]))

    frame = 0
    while True:
        ret_val, image = cam.read()

        logger.debug("image process+")
        humans = e.inference(
            image,
            resize_to_default=(w > 0 and h > 0),
            upsample_size=args.resize_out_ratio,
        )

        logger.debug("postprocess+")

        image = TfPoseEstimator.draw_humans(
            image, humans, imgcopy=False, frame=frame, output_json_dir=args.output_json
        )
        frame += 1

        logger.debug("show+")
        cv2.putText(
            image,
            "FPS: %f" % (1.0 / (time.time() - fps_time)),
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.imshow("tf-pose-estimation result", image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug("finished+")

    cv2.destroyAllWindows()
