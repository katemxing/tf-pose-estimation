import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from tqdm import tqdm

logger = logging.getLogger("TfPoseEstimator-Video")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tf-pose-estimation Video")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument(
        "--resolution",
        type=str,
        default="432x368",
        help="network input resolution. default=432x368",
    )
    parser.add_argument(
        "--model", type=str, default="mobilenet_thin", help="cmu / mobilenet_thin"
    )
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
        "--show-process",
        type=bool,
        default=False,
        help="for debug purpose, if enabled, speed for inference is dropped.",
    )
    parser.add_argument(
        "--showBG", type=bool, default=True, help="False to show skeleton only."
    )
    parser.add_argument(
        "--output_json", type=str, default="/tmp/", help="writing output json dir"
    )
    args = parser.parse_args()

    logger.debug("initialization %s : %s" % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    def count_frames(video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release the video capture object
        cap.release()

        return total_frames

    total_frames = count_frames(args.video)
    print(f"Total frames in the video: {total_frames}")

    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    # To avoid exception:
    # frame = 0
    # while cap.isOpened():

    for frame in tqdm(range(int(total_frames))):
        ret_val, image = cap.read()

        humans = e.inference(
            image,
            resize_to_default=(w > 0 and h > 0),
            upsample_size=args.resize_out_ratio,
        )

        # print(humans.upsample_size)

        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(
            image, humans, imgcopy=False, frame=frame, output_json_dir=args.output_json
        )
        # print("frame=", frame)
        # frame += 1

        # Uncomment the following if GUI is needed:
        # cv2.putText(
        #     image,
        #     "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #     (10, 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     2,
        # )
        # cv2.imshow("tf-pose-estimation result", image)
        # fps_time = time.time()
        # if cv2.waitKey(1) == 27:
        #     break

    # cv2.destroyAllWindows()
logger.debug("finished+")
