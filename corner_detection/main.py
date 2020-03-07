"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 main.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 main.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 main.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 main.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 main.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import pdb
import numpy as np
from PIL import Image, ImageDraw
import skimage.io
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".


from config import Config
import model.utils as utils
import model.model as modellib
import scipy.misc
from random import randint
import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained/mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DATASET_BASE_DIR = '/local-scratch/fuyang/cities_dataset/'
IMAGE_RGB_DIR = os.path.join(DATASET_BASE_DIR, 'rgb')

INCLUDE_EDGE = False

############################################################
#  Configurations
############################################################

class BuildingsConfig(Config):

    # Give the configuration a recognizable name
    NAME = "outdoorDataset"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 3  # we have bg, edge and corner


class InferenceConfig(BuildingsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################

class BuildingsDataset(utils.Dataset):
    def load_buildings(self, phase):
        self.phase = phase
        # Add classes
        if phase != 'train' and phase != 'test':
            raise ValueError('Invalid phase {} for BuildingDataset'.format(phase))

        if INCLUDE_EDGE:
            self.add_class("buildings", 1, "edge")
            self.add_class("buildings", 2, "corner")
        else:
            self.add_class("buildings", 2, "corner")

        # Add images
        rgb_prefix = os.path.join(DATASET_BASE_DIR, 'rgb')
        if self.phase == 'train':
            train_path = os.path.join(DATASET_BASE_DIR, 'train_list.txt')
            with open(train_path) as f:
                train_list = f.readlines()

            for k, im_id in enumerate(train_list):
                im_path = os.path.join(rgb_prefix, im_id.strip()+'.jpg')
                for i in range(4):
                    self.add_image("buildings", randint(0, 359), False, image_id=8*k+2*i, path=im_path)
                    self.add_image("buildings", randint(0, 359), True, image_id=8*k+2*i+1, path=im_path)
        elif self.phase == 'test':
            test_path = os.path.join(DATASET_BASE_DIR, 'valid_list.txt')
            with open(test_path) as f:
                test_list = f.readlines()

            for i, im_id in enumerate(test_list):
                im_path = os.path.join(rgb_prefix, im_id.strip()+'.jpg')
                self.add_image("buildings", 0, False, image_id=i, path=im_path)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        im_path = info['path']
        if self.phase == 'train':
            rot = info['rot']

        # load annotations
        p_path = im_path.replace('.jpg', '.npy').replace('rgb', 'annot')  # use the new annotations
        v_set = np.load(open(p_path, 'rb'),  encoding='bytes', allow_pickle=True)
        seg_path = im_path.replace('rgb', 'outline')
        seg_img = Image.fromarray(np.zeros((256, 256)).astype('uint8')) #Image.open(seg_path)
        graph = dict(v_set[()])

        # draw mask
        masks, class_ids, edges_dir, coords = [], [], [], []

        # prepare edge instances for this image
        edge_set = set()
        for v1 in graph:
            for v2 in graph[v1]:
                x1, y1 = v1
                x2, y2 = v2
                # make an order
                if x1 > x2:
                    x1, x2, y1, y2 = x2, x1, y2, y1  # swap
                elif x1 == x2 and y1 > y2:
                    x1, x2, y1, y2 = x2, x1, y2, y1  # swap
                else:
                    pass
                edge = (x1, y1, x2, y2)
                edge_set.add(edge)

        # debug
        # import matplotlib.pyplot as plt
        # print(rot)
        # print(info['flip'])

        # rgb_im = Image.open(im_path).rotate(rot)
        # if info['flip']:
        #     rgb_im = rgb_im.transpose(Image.FLIP_LEFT_RIGHT)
        # draw_debug = ImageDraw.Draw(rgb_im)
        # debug
        if self.phase == 'train':
            seg_img = seg_img.rotate(rot)
            if info['flip']:
                seg_img = seg_img.transpose(Image.FLIP_LEFT_RIGHT)

        if INCLUDE_EDGE:
            edge_list = list(edge_set)
            for edge in edge_list:
                x1, y1, x2, y2 = edge
                # create mask
                mask_im = Image.fromarray(np.zeros((256, 256)))

                # draw lines
                draw = ImageDraw.Draw(mask_im)
                draw.line((x1, y1, x2, y2), fill='white', width=4)

                # apply augmentation
                if self.phase == 'train':
                    mask_im = mask_im.rotate(rot)
                    x1, y1 = self.rotate_coords(np.array([256, 256]), np.array([x1, y1]), rot)
                    x2, y2 = self.rotate_coords(np.array([256, 256]), np.array([x2, y2]), rot)

                    if info['flip']:
                        mask_im = mask_im.transpose(Image.FLIP_LEFT_RIGHT)
                        x1, y1 = (128-abs(128-x1), y1) if x1 > 128 else (128+abs(128-x1), y1)
                        x2, y2 = (128-abs(128-x2), y2) if x2 > 128 else (128+abs(128-x2), y2)

                # import matplotlib.pyplot as plt
                # plt.imshow(mask_im)
                # plt.show()

                # compute angle
                pc = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
                pp = np.array([0, 1])
                pr = np.array([x1, y1]) if x1 >= x2 else np.array([x2, y2])
                pr -= pc
                cosine_angle = np.dot(pp, pr) / (np.linalg.norm(pp) * np.linalg.norm(pr))
                if np.isnan(cosine_angle) or np.isinf(cosine_angle):
                    pdb.set_trace()

                # angle = np.arccos(cosine_angle)
                # angle = 180.0 - np.degrees(angle)
                angle = 0 # DUMMY VALUE 
                delta_degree = 10.0
                n_bins = 18
                bin_num = 0 #(int(angle/delta_degree+0.5)%n_bins)

                #print(bin_num, angle)
                if np.sum(mask_im) > 0:
                    edges_dir.append(bin_num)
                    masks.append(np.array(mask_im))
                    class_ids.append(1)
                    coords.append([y1, x1, y2, x2])

            # # draw edge debug
            # draw_debug.line((x1, y1, x2, y2), fill='blue', width=2)

            # print(rot)
            # print(info['flip'])
            # print((int(angle/30+0.5)%6))
            # import matplotlib.pyplot as plt

            # draw.line((x1, y1, x2, y2), fill='blue', width=2)

            # plt.imshow(rgb_im)
            # plt.show()

        # prepare corner instances for this image
        for v in graph:
            mask_im = Image.fromarray(np.zeros((256, 256)))
            # draw circles
            x, y = v
            draw = ImageDraw.Draw(mask_im)
            draw.ellipse([x-4, y-4, x+4, y+4], fill='white', outline='white')

            # apply augmentation during training
            if self.phase == 'train':
                mask_im = mask_im.rotate(rot)
                x, y = self.rotate_coords(np.array([256, 256]), np.array([x, y]), rot)

                if info['flip']:
                    mask_im = mask_im.transpose(Image.FLIP_LEFT_RIGHT)
                    x, y = (128-abs(128-x), y) if x > 128 else (128+abs(128-x), y)

            if np.sum(mask_im) > 0: 
                edges_dir.append(-1)
                masks.append(np.array(mask_im))
                class_ids.append(2)
                coords.append([y, x, -1, -1])

        
        seg_arr = np.array(seg_img)/255.0
        masks = np.stack(masks).astype('float').transpose(1, 2, 0)
        class_ids = np.array(class_ids).astype('int32')
        edges_dir = np.array(edges_dir)
        coords = np.array(coords)
        
        # # debug
        # deb_mask = np.sum(masks, -1)
        # rgb_im = Image.open(im_path).rotate(rot)
        # if info['flip']:
        #     rgb_im = rgb_im.transpose(Image.FLIP_LEFT_RIGHT)
        # draw = ImageDraw.Draw(rgb_im)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(rgb_im)
        # plt.figure()
        # plt.imshow(deb_mask)
        # plt.show()

        # For debugging purpose, visualizing the edge we drew in a single image
        # mask_im_all_edge = Image.fromarray(np.zeros((256, 256)))
        # mask_im_all_corner = Image.fromarray(np.zeros((256, 256)))
        # for edge in edge_list:
        #     x1, y1, x2, y2 = edge
        #     x1, y1 = self.rotate_coords(np.array([256, 256]), np.array([x1, y1]), rot)
        #     x2, y2 = self.rotate_coords(np.array([256, 256]), np.array([x2, y2]), rot)
        #     if info['flip']:
        #         x1, y1 = (128-abs(128-x1), y1) if x1 > 128 else (128+abs(128-x1), y1)
        #         x2, y2 = (128-abs(128-x2), y2) if x2 > 128 else (128+abs(128-x2), y2)

        #     draw = ImageDraw.Draw(mask_im_all_edge)
        #     draw.line((x1, y1, x2, y2), fill='white', width=4)


        # for v in graph:
        #     x, y = v
        #     draw = ImageDraw.Draw(mask_im_all_corner)
        #     draw.ellipse([x-1, y-1, x+1, y+1], fill='white', outline='white')
        #
        # scipy.misc.imsave('./mask-all-corner.jpg', mask_im_all_corner)
        # scipy.misc.imsave('./mask-all-edge.jpg', mask_im_all_edge)

        return masks, class_ids, edges_dir, coords, seg_arr

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "buildings":
            return info["buildings"]
        else:
            super(BuildingsDataset, self).image_reference(image_id)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        im = Image.open(info['path'])
        # dp_im = Image.open(info['path'].replace('rgb', 'depth')).convert('L')
        # surf_im = Image.open(info['path'].replace('rgb', 'surf'))
        # gray_im = Image.open(info['path'].replace('rgb', 'gray')).convert('L')
        # out_im = Image.open(info['path'].replace('rgb', 'outlines')).convert('L')

        if self.phase == 'train':
            rot = info['rot']
            im = im.rotate(rot)
            # dp_im = dp_im.rotate(rot)
            # surf_im = surf_im.rotate(rot)
            # gray_im = gray_im.rotate(rot)
            # out_im = out_im.rotate(rot)

            if info['flip'] == True:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                # dp_im = dp_im.transpose(Image.FLIP_LEFT_RIGHT)
                # surf_im = surf_im.transpose(Image.FLIP_LEFT_RIGHT)
                # gray_im = gray_im.transpose(Image.FLIP_LEFT_RIGHT)
                # out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)
        #out_im = np.array(out_im)

        xs = np.tile((np.arange(0, 256)[np.newaxis, :]), (256, 1))
        ys = xs.transpose(1, 0)
        xs = xs[:, :, np.newaxis]
        ys = ys[:, :, np.newaxis]

        #return np.concatenate([np.array(im), np.array(dp_im)[:, :, np.newaxis], np.array(gray_im)[:, :, np.newaxis], np.array(surf_im)], axis=-1)
        #return np.concatenate([np.array(im), np.array(dp_im)[:, :, np.newaxis], np.array(out_im)[:, :, np.newaxis]], axis=-1)
        return np.array(im)

    def rotate_coords(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        return new+rot_center

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BuildingsConfig()
    else:
        class InferenceConfig(BuildingsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    torch.manual_seed(config.SEED)

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs, input_channel=3, corner_only=not INCLUDE_EDGE)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs, input_channel=3, corner_only=not INCLUDE_EDGE)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.command == 'train' and args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
        # Load weights
        print("Loading weights for training from {}".format(model_path))
        model.load_pretrained_weights(model_path, extra_channels=2)
    else:
        model_path = ""

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = BuildingsDataset()
        dataset_train.load_buildings("train")
        dataset_train.prepare()

        dataset_test = BuildingsDataset()
        dataset_test.load_buildings("test")
        dataset_test.prepare()

        # # Validation dataset
        # dataset_val = BuildingsDataset()
        # dataset_val.load_buildings(args.dataset, "minival")
        # dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # # Training - Stage 1
        # print("Training network heads")
        # model.train_model(dataset_train, dataset_train,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=20,
        #             layers='heads')

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train_model(dataset_train, dataset_train,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=40,
        #             layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_test,
                    learning_rate=config.LEARNING_RATE / 5,
                    epochs=20,
                    layers='all')

        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_test,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=60,
                    layers='all')

        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_test,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=100,
                    layers='all')

    elif args.command == "evaluate":
        # Load weights trained on MS-COCO
        last_saved = './logs/la_dataset20181021T2225_corner/mask_rcnn_la_dataset_0060.pth'
        model.load_state_dict(torch.load(last_saved))
        class_names = ['BG', 'edge', 'corner']

        # Validation dataset
        dataset_test = BuildingsDataset()
        dataset_test.load_buildings('test')
        dataset_test.prepare()

        im_path = os.path.join(DATASET_BASE_DIR, 'valid_list.txt')
        with open(im_path) as f:
            im_path_list = [x.strip()+'.jpg' for x in f.readlines()]

        im_list_rgb = [skimage.io.imread(os.path.join(IMAGE_RGB_DIR, path)) for path in im_path_list]
        im_list_depth = [np.array(Image.open(os.path.join(IMAGE_DEPTH_DIR, path)).convert('L')) for path in im_path_list]
        im_list_outline = [np.array(Image.open(os.path.join(IMAGE_OUTLINE_DIR, path)).convert('L')) for path in im_path_list]
        im_list_surf = [np.array(Image.open(os.path.join(IMAGE_SURF_DIR, path))) for path in im_path_list]
        im_list_gray = [np.array(Image.open(os.path.join(IMAGE_GRAY_DIR, path)).convert('L')) for path in im_path_list]
        model.evaluate_map(test_dataset=dataset_test, image_list=im_list_rgb, vocabulary=class_names, with_depth=True, image_list_depth=im_list_depth, image_list_gray=im_list_gray, image_list_surf=im_list_surf, image_list_outline=im_list_outline)
        # model.evaluate_map(test_dataset=dataset_test, image_list=im_list_rgb, vocabulary=class_names)

        print("Finish evaluation")

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
