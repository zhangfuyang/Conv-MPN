# Conv-MPN: Convolutional Message Passing Neural Network for Structured Outdoor Architecture Reconstruction

## Overview

This is the official implementation of the paper [Conv-MPN: Convolutional Message Passing Neural Network for Structured Outdoor Architecture Reconstruction](https://arxiv.org/abs/1912.01756).

This paper proposes a novel message passing neural (MPN) architecture Conv-MPN, which reconstructs an outdoor building as a planar graph from a single RGB image.

If you find the paper and the code helpful, please consider citing this paper:

```
@article{conv-mpn,
Title = {Conv-MPN: Convolutional Message Passing Neural Network for Structured Outdoor Architecture Reconstruction},
Author = {Fuyang, Zhang and Nelson, Nauata and Yasutaka, Furukawa},
Year = {2019},
journal = {arXiv preprint arXiv:1912.01756}
}
```

## Environment Setup

The implementation is based on Python3.5 and Pytorch1.2. You can do `pip install -r requirements.txt` to install the dependencies.

## Data Download
Please download data from this [link](https://www.dropbox.com/sh/q1jmqnm26q21h1a/AABtxO0Uni9eZs-Qs37HJTJLa?dl=0). The dataset we used is from the paper ["Vectorizing World Buildings: Planar Graph Reconstruction by Primitive Detection and Relationship Classification"](https://arxiv.org/abs/1912.05135).

## Data preprocessing
**Corner detection**: Given an input RGB image, we first use mask-rcnn to detect corner candidates. We treat each corner as 8x8 bounding box with a corner at the center. In the training, building graphs are generated from annotation and detected corners. In the testing, we only use detected corners. See `./corner_detection/README.md` for information related to installation and environment setup.

1. run `./corner_detection/main.py` to train the corner detector.
2. run `./corner_detection/demo.py` to generate detected corners for the dataset.
3. Note that the above instructions are for processing data with annotation files. If the aim is to run pre-trained Conv-MPN, you can just simply download the corner detection results [here](https://drive.google.com/open?id=1AikD1FlNfpFqJy3g-mx8V2KtpYKDzV0Q) and pretrained model [here](https://drive.google.com/open?id=1ov-9ejGWpzUy0-M2hGiPgv4bK5hlR26I).


## Conv-MPN

### Training

To train the Network from scratch, please run:
```
python train.py
```

You can modify some settings in the `config.py`.

1. Conv-MPN is GPU memory intensive due to the use of 3D feature volumes. We use two NVIDIA TitanX GPUs (24G RAM each) for training Conv-MPN(t=3). t=3 means 3 times convolutional message passing. You can specify the gpu devices in `config.py` script.
2. `config.py` contains a variable `model_loop_time`, which indicates how many iterations of message passing. In our experiments, 3 times achieves best score.
3. variables `conv_mpn` and `gnn` are used for setting the network modes.
	1. conv_mpn = true, gnn = false means using conv-mpn network
	2. conv_mpn = false, gnn = true means using gnn network
	3. conv_mpn = false, gnn = false means per edge classification network.

We compared all the experiments in the paper, please see Experiemnt Section in the paper for the details.

### Testing

To evaluate the performance or our trained model, please run:
```
python demo.py
```

You can download pretrained model from [here](https://drive.google.com/open?id=1CEQd2kSTMwlgKyyrKDdqBmJbqUCmnNFq).


# Contact
If you have any questions, please contact me at fuyang.zhang97@gmail.com or fuyangz@sfu.ca
