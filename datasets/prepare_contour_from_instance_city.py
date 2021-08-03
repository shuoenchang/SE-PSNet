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
import glob
from PIL import Image


def _process_instance_to_semantic(img_path, output_path):

    laplacian_kernel = torch.tensor(
        [-1, -1, -1,
         -1,  8, -1,
         -1, -1, -1],
        dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)
    img = Image.open(img_path)
    img = np.array(img)
    # print(img_path)
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
    output_img = output_path.replace('contour_train', 'contour_train_img').replace('npz', 'png')
    
    # cv2.imshow("targets_contour", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # save as compressed npz
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.replace('contour_train', 'contour_train_img'), exist_ok=True)
    cv2.imwrite(output_img, output*255)
    np.savez_compressed(output_path, mask=output)
    # Image.fromarray(output).save(output_semantic)


def create_coco_contour_from_semantic(data_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    instance_gt_paths = glob.glob(os.path.join(data_root, "*", "*_gtFine_instanceIds.png"))

    def iter_annotations():
        for gt_path in tqdm(instance_gt_paths):
            img_path = gt_path
            output = gt_path.replace('gtFine_instanceIds.png', 'leftImg8bit.npz').replace('train', 'contour_train')
            yield img_path, output

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
        default="cityscapes",
        help="dataset to generate",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    dataset_dir = os.path.join(os.path.dirname(__file__), args.dataset_name)
    for s in ["train2017"]:
        create_coco_contour_from_semantic(
            os.path.join(dataset_dir, "gtFine", "train"),
            os.path.join(dataset_dir, "gtFine", "contour_train")
        )
