import argparse
import cv2
import numpy as np
import time
import logging

import tensorflow as tf

from common import CocoPairsRender, CocoColors, preprocess, estimate_pose, draw_humans
from network_cmu import CmuNetwork
from network_mobilenet import MobilenetNetwork
from networks import get_network
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


fps_time = 0


def cb_showimg(img, preprocessed, heatMat, pafMat, humans, show_process=False):
    global fps_time

    # display
    image = img
    image_h, image_w = image.shape[:2]
    image = draw_humans(image, humans)

    scale = 480.0 / image_h
    newh, neww = 480, int(scale * image_w + 0.5)

    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

    if show_process:
        process_img = CocoPoseLMDB.display_image(preprocessed, heatMat, pafMat, as_numpy=True)
        process_img = cv2.resize(process_img, (640, 480), interpolation=cv2.INTER_AREA)

        canvas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
        canvas[:, :640] = process_img
        canvas[:, 640:] = image
    else:
        canvas = image

    cv2.putText(canvas, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('openpose', canvas)

    fps_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Realtime Webcam')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='mobilenet', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    parser.add_argument('--show-process', type=bool, default=False, help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session() as sess:
        net, _, last_layer = get_network(args.model, input_node, sess)

        cam = cv2.VideoCapture(args.camera)
        ret_val, img = cam.read()
        logging.info('cam image=%dx%d' % (img.shape[1], img.shape[0]))

        while True:
            logging.debug('cam read+')
            ret_val, img = cam.read()

            logging.debug('cam preprocess+')
            if args.zoom < 1.0:
                canvas = np.zeros_like(img)
                img_scaled = cv2.resize(img, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
                dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
                canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
                img = canvas
            elif args.zoom > 1.0:
                img_scaled = cv2.resize(img, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (img_scaled.shape[1] - img.shape[1]) // 2
                dy = (img_scaled.shape[0] - img.shape[0]) // 2
                img = img_scaled[dy:img.shape[0], dx:img.shape[1]]
            preprocessed = preprocess(img, args.input_width, args.input_height)

            logging.debug('cam process+')
            pafMat, heatMat = sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [preprocessed]}
            )
            heatMat, pafMat = heatMat[0], pafMat[0]

            logging.debug('cam postprocess+')
            t = time.time()
            humans = estimate_pose(heatMat, pafMat)

            logging.debug('cam show+')
            cb_showimg(img, preprocessed, heatMat, pafMat, humans, show_process=args.show_process)

            if cv2.waitKey(1) == 27:
                break  # esc to quit
            logging.debug('cam finished+')
    cv2.destroyAllWindows()
