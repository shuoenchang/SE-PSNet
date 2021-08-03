# SE-PSNet: Silhouette-based Enhancement Feature for Panoptic Segmentation Network [[arxiv](https://arxiv.org/abs/2107.05093)]
This project hosts the code for implementing the SE-PSNet for panoptic segmentation.


## Installation
This implementation uses Python 3.7, [Pytorch1.8.0](http://pytorch.org/), cudatoolkit 11.1. We recommend to use [conda](https://docs.conda.io/en/latest/miniconda.html) to deploy the environment.

* Install environment:
```Shell
conda create -n sepsnet python=3.7
conda activate sepsnet
bash init.sh
```


## Demo
The model of COCO can be download [here](https://drive.google.com/file/d/1-U4oBvULEM0shsTV7kp0FSuT-kgaFzOY/), put the model under `weights/`.

* Demo on images
  ```Shell
  python demo/demo.py --config configs/COCO_R_101.yaml --confidence 0.3 --input $IMAGE_PATH --output $OUTPUT_PATH --opts MODEL.WEIGHTS weights/COCO_R_101.pth
  ```

* Demo on videos
  ```Shell
  python demo/demo.py --config configs/COCO_R_101.yaml --confidence 0.3 --video-input $VIDEO_PATH --output $OUTPUT_PATH --opts MODEL.WEIGHTS weights/COCO_R_101.pth
  ```

* Demo with webcam
  ```Shell
  python demo/demo.py --config configs/COCO_R_101.yaml --confidence 0.3 --webcam --opts MODEL.WEIGHTS weights/COCO_R_101.pth
  ```

The model of CityScapes can be download [here](https://drive.google.com/file/d/1wO_2FZpXHYo9GPxzih1-W8UYriIagC45/), put the model under `weights/`.

* Demo on images
  ```Shell
  python demo/demo.py --config configs/CityScapes_R_101.yaml --confidence 0.3 --input $IMAGE_PATH --output $OUTPUT_PATH --opts MODEL.WEIGHTS weights/CityScapes_R_101.pth
  ```

* Demo on videos
  ```Shell
  python demo/demo.py --config configs/CityScapes_R_101.yaml --confidence 0.3 --video-input $VIDEO_PATH --output $OUTPUT_PATH --opts MODEL.WEIGHTS weights/CityScapes_R_101.pth
  ```


## Data preparation
### COCO
-  Download the [COCO Panoptic dataset](https://cocodataset.org/#download) and the panoptic annotations  
- Place them like the following structures. 
  ```
    datasets
    ├── coco 
    │   ├── annotations
    │   │   ├── panoptic_{train, val}2017.json
    │   │   ├── instances_{train, val}2017.json
    │   ├── panoptic_{train,val}2017 # panoptic GT image
    │   ├── {train, val, test}2017 # image files
  ```
- Data pre-process
  ```Shell
  cd datasets
  python prepare_panoptic_fpn.py # extract semantic annotations from panoptic annotations
  python prepare_thing_sem_from_instance.py # extract semantic labels from instance annotations
  python prepare_contour_from_instance.py # produce silhouette gt for thing
  python prepare_contour_from_stuff.py # produce silhouette gt for stuff
  ```


## Training and Testing
We use the configuration file (see 'configs/****.yaml') to fully control the training/testing process.
- Make sure to download the entire dataset and do pre-process using the commands above.
- Download the pre-train model(FCOS_MS_R_101_2x.pth) from [FCOS](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/FCOS-Detection/README.md) and put it in `weights/`.
- Run the command below for training.
  ```Shell
  python tools/train_net.py --config configs/COCO_R_101.yaml --resume
  ```
- Run the command below for validation.
  ```Shell
  python tools/train_net.py --config configs/COCO_R_101.yaml --resume --eval
  ```
- Run the command below for test-dev.
  ```Shell
  python tools/train_net.py --config configs/COCO_R_101.yaml --resume --eval DATASETS.TEST coco_2017_test-dev_panoptic_separated
  ```

## Citation
If you use SE-PSNet or this code base in your work, please cite
```
@misc{chang2021sepsnet,
    title={SE-PSNet: Silhouette-based Enhancement Feature for Panoptic Segmentation Network},
    author={Shuo-En Chang and Yi-Cheng Yang and En-Ting Lin and Pei-Yung Hsiao and Li-Chen Fu},
    year={2021},
    eprint={2107.05093},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

On the other hand, our work is based on the following works. Thanks for their hard-working.

- [Detectron2](https://github.com/facebookresearch/detectron2)
  ```
  @misc{wu2019detectron2,
    author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                    Wan-Yen Lo and Ross Girshick},
    title =        {Detectron2},
    howpublished = {\url{https://github.com/facebookresearch/detectron2}},
    year =         {2019}
  }
  ```

- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
  ```
  @misc{tian2019adelaidet,
    author =       {Tian, Zhi and Chen, Hao and Wang, Xinlong and Liu, Yuliang and Shen, Chunhua},
    title =        {{AdelaiDet}: A Toolbox for Instance-level Recognition Tasks},
    howpublished = {\url{https://git.io/adelaidet}},
    year =         {2019}
  }
  ```