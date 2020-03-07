import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import main
import model.utils as utils
import model.model as modellib
import model.visualize as visualize
from PIL import Image, ImageDraw
import torch
import pdb
import svgwrite

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
DATASET_BASE_DIR = '/local-scratch/fza49/cities_dataset/'
IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'rgb')
# = os.path.join(DATASET_BASE_DIR, 'depth')
# IMAGE_SURF_DIR = os.path.join(DATASET_BASE_DIR, 'surf')
# IMAGE_GRAY_DIR = os.path.join(DATASET_BASE_DIR, 'gray')
#IMAGE_OUTLINE_DIR = os.path.join(DATASET_BASE_DIR, 'outline')

def compute_angle(edge):

    # compute angle
    y2, x2, y1, x1 = edge
    pc = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    pp = np.array([0, 1])
    pr = np.array([x1, y1]) if x1 >= x2 else np.array([x2, y2])
    pr -= pc
    cosine_angle = np.dot(pp, pr) / (np.linalg.norm(pp) * np.linalg.norm(pr) + 1E-8)
    angle = np.arccos(cosine_angle)
    angle = 180.0 - np.degrees(angle)

    delta_degree = 10.0
    n_bins = 18
    bin_num = (int(angle/delta_degree+0.5)%n_bins)

    return bin_num

class InferenceConfig(main.BuildingsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config, input_channel=3, corner_only=not main.INCLUDE_EDGE)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
saved_model = '/local-scratch/fza49/nnauata/building_reconstruction/geometry-primitive-detector/logs/trainingdoubleset220190903T1533/mask_rcnn_trainingdoubleset2_0001.pth'
model.load_state_dict(torch.load(saved_model))

# _, last_saved = model.find_last()
# model.load_state_dict(torch.load(last_saved))
print('loaded weights from {}'.format(saved_model))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'edge', 'corner']

# Load a random image from the images folder
im_path = '/local-scratch/fza49/cities_dataset/all_list.txt'
dst_path = '/local-scratch/fza49/test'
with open(im_path) as f:
    im_list = [x.strip()+'.jpg' for x in f.readlines()]
file_names = im_list

for fname in file_names:
    for rot in [0]:  #, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:#, 90, 180, 270]: #, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        for flip in [False]:

            _id = fname.replace('.jpg', '')
            im = Image.open(os.path.join(IMAGE_DIR, fname))
            #dp_im = Image.open(os.path.join(IMAGE_DEPTH_DIR, fname)).convert('L')
            # surf_im = Image.open(os.path.join(IMAGE_SURF_DIR, fname))
            # gray_im = Image.open(os.path.join(IMAGE_GRAY_DIR, fname)).convert('L')
            #out_im = Image.open(os.path.join(IMAGE_OUTLINE_DIR, fname)).convert('L')

            # Rotate images
            im = im.rotate(rot)
            #dp_im = dp_im.rotate(rot)
            # surf_im = surf_im.rotate(rot)
            # gray_im = gray_im.rotate(rot)
            #out_im = out_im.rotate(rot)

            # Flip images
            if flip == True:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                #dp_im = dp_im.transpose(Image.FLIP_LEFT_RIGHT)
                # surf_im = surf_im.transpose(Image.FLIP_LEFT_RIGHT)
                # gray_im = gray_im.transpose(Image.FLIP_LEFT_RIGHT)
                #out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)

            gt_class_ids, gt_coords = utils.generate_boxes_from_gt(os.path.join(IMAGE_DIR, fname), rot, flip)
            edges_dir = np.array([compute_angle(coord) if cl == 1 else -1 for cl, coord in zip(gt_class_ids, gt_coords)])
            gt_class_ids = torch.from_numpy(gt_class_ids).cuda()
            gt_coords = torch.from_numpy(gt_coords).cuda()
            edges_dir = torch.from_numpy(edges_dir).cuda()

            # Convert to array
            im = np.array(im)
            #dp_im = np.array(dp_im)[:, :, np.newaxis]
            # surf_im = np.array(surf_im)
            # gray_im = np.array(gray_im)[:, :, np.newaxis]
            #out_im = np.array(out_im)[:, :, np.newaxis]

            # Run detection
            xs = np.tile((np.arange(0, 256)[np.newaxis, :]), (256, 1))
            ys = xs.transpose(1, 0)
            xs = xs[:, :, np.newaxis]
            ys = ys[:, :, np.newaxis]
            #input_im = np.concatenate([im, out_im], axis=-1)
            #input_im = np.concatenate([im, dp_im, xs, ys], axis=-1)
            results, outline = model.detect([im], gt_class_ids, gt_coords, edges_dir)

            # Visualize results
            r = results[0]

            # Grab corners and edges
            edges = r['rois'][r['class_ids'] == 1, :]
            corners = r['rois'][r['class_ids'] == 2, :]
            edges_dir = r['edges_dir'][r['class_ids'] == 1, :]
            corners_feat = r['corners_emb']
            edges_feat = r['edges_emb']
            edges_conf = r['scores'][r['class_ids'] == 1]

            # Extract edges
            e_coords = []
            for j, e, d in zip(range(edges.shape[0]), edges, edges_dir):
                y1, x1, y2, x2 = utils.edge_endpoints(e, d)
                e_coords.append([y1, x1, y2, x2])

            # Filter out low confidence edges
            # e_coords = np.array(e_coords)
            # e_coords = e_coords[edges_conf > 0.2, :]
            # edges_feat = edges_feat[edges_conf > 0.2, :]

            # Extract corners
            c_coords = []
            for i, c in enumerate(corners):
                y1, x1, y2, x2 = c
                yc = (y2+y1)/2.0
                xc = (x2+x1)/2.0
                c_coords.append([yc, xc, -1, -1])

            # Save
            dest = 'annots_only'
            path = "{}/{}/corners/".format(dst_path, dest)
            if not os.path.exists(path):
                os.makedirs(path)
            path = "{}/{}/edges/".format(dst_path, dest)
            if not os.path.exists(path):
                os.makedirs(path)
            # path = "{}/{}/corners_feats/".format(dst_path, dest)
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # path = "{}/{}/edges_feats/".format(dst_path, dest)
            # if not os.path.exists(path):
            #     os.makedirs(path)

            # todo: note that the edges here are wrong, becasue the dir_prediction are disabled durnig inference
            # from scipy.misc import imsave
            # import cv2
            # for i in range(edges_feat.shape[0]):
            #     x = edges_feat[i, :3]
            #     x = np.transpose(x, [1, 2, 0]) + np.array([123.7, 116.8, 103.9])
            #
            #     imsave('test_{}.png'.format(i), x)
            #     cv2.line(im, (int(e_coords[i][1]), int(e_coords[i][0])), (int(e_coords[i][3]), int(e_coords[i][2])), (255,0,0),2)
            #     imsave('test_img.png', im)

            np.save("{}/{}/corners/{}_{}_{}.npy".format(dst_path, dest, _id, rot, flip), c_coords)
            np.save("{}/{}/edges/{}_{}_{}.npy".format(dst_path, dest, _id, rot, flip), e_coords)
            # np.save("{}/{}/corners_feats/{}_{}_{}.npy".format(dst_path, dest, _id, rot, flip), corners_feat)
            # np.save("{}/{}/edges_feats/{}_{}_{}.npy".format(dst_path, dest, _id, rot, flip), edges_feat)
