# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import functools
import multiprocessing as mp
import os
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch import nn
from torch.nn import functional as F


def _process_instance_to_semantic(output_path, img_path):

    laplacian_kernel = torch.tensor(
        [-1, -1, -1,
         -1,  8, -1,
         -1, -1, -1],
        dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img[img == 255] = 0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    contour = F.conv2d(img, laplacian_kernel, padding=1)
    contour = contour.squeeze()
    H, W = contour.shape
    contour[contour != 0] = 1
    contour[contour == 0] = 0
    contour[0, :] = 0
    contour[:, 0] = 0
    contour[H-1, :] = 0
    contour[:, W-1] = 0
    output = contour.numpy()
    # output_img = output_path.replace('contour_stuff_train2017', 'contour_stuff_train2017_img').replace('npz', 'png')
    
    # cv2.imshow("targets_contour", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # save as compressed npz
    # cv2.imwrite(output_img, output*255)
    np.savez_compressed(output_path, mask=output)
    # Image.fromarray(output).save(output_semantic)


def create_coco_contour_from_semantic(stuff_gt_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    stuff_gt_paths = os.listdir(stuff_gt_root)

    def iter_annotations():
        for gt_path in stuff_gt_paths:
            img_path = os.path.join(stuff_gt_root, gt_path)
            output = os.path.join(output_root, gt_path + '.npz').replace('.png', '')
            yield output, img_path

    # single process
    # print("Start writing to {} ...".format(output_root))
    # start = time.time()
    # for output, img in iter_annotations():
    #     _process_instance_to_semantic(output, img)
    # print("Finished. time: {:.2f}s".format(time.time() - start))

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    print("Start writing to {} ...".format(output_root))
    start = time.time()
    pool.starmap(
        functools.partial(
            _process_instance_to_semantic),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))

    return


def get_parser():
    parser = argparse.ArgumentParser(description="Keep only model in ckpt")
    parser.add_argument(
        "--dataset-name",
        default="coco",
        help="dataset to generate",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    dataset_dir = os.path.join(os.path.dirname(__file__), args.dataset_name)
    for s in ["train2017"]:
        create_coco_contour_from_semantic(
            os.path.join(dataset_dir, "panoptic_stuff_{}".format(s)),
            os.path.join(dataset_dir, "contour_stuff_{}".format(s))
        )
