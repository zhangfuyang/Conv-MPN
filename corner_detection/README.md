# geometry-primitive-detector

This is a detector adapted from the [pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn). 
It detects geometric primitives such as edges/polygonal regions in input images by giving bounding boxes and segentation masks, it can also do corner detection by producing a corner heatmap.

Please follow the README of the original pytorch-mask-rcnn repo(we remove the stuffs that we don't use and attach useful instructions below) to finish the compilation and other preparation works.


----------

This is a Pytorch implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) that is in large parts based on Matterport's
[Mask_RCNN](https://github.com/matterport/Mask_RCNN). Matterport's repository is an implementation on Keras and TensorFlow.


## Requirements
* Python 3
* Pytorch 0.4.0
* matplotlib, scipy, skimage, h5py, it's better to update these libs to the latest versions

## Installation
1. Clone this repository.

    
2. We use functions from two more repositories that need to be build with the right `--arch` option for cuda support.
The two functions are Non-Maximum Suppression from ruotianluo's [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
repository and longcw's [RoiAlign](https://github.com/longcw/RoIAlign.pytorch).

    | GPU | arch |
    | --- | --- |
    | TitanX | sm_52 |
    | GTX 960M | sm_50 |
    | GTX 1070 | sm_61 |
    | GTX 1080 (Ti) | sm_61 |

        cd nms/src/cuda/
        nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../

        cd roialign/roi_align/src/cuda/
        nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../../
    
3. Download the pretrained models on COCO and ImageNet from [Google Drive](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).

## Demo

To test your installation simply run the demo with

    python demo.py



## Training
Training and evaluation code is in main.py. You can run it from the command
line as such:

    # Train a new model starting from pre-trained COCO weights
    python main.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python main.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python main.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained. This will find
    # the last trained weights in the model directory.
    python main.py train --dataset=/path/to/coco/ --model=last

For evaluation:

    # Run valuation on the last trained model
    python main.py evaluate

The training schedule, learning rate, and other parameters can be set in main.py.

## Results


