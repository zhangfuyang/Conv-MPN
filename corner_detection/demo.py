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
DATASET_BASE_DIR = '/local-scratch/fuyang/cities_dataset/'
IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'rgb')
SAVE_DIR = os.path.join(ROOT_DIR, 'result')

class InferenceConfig(main.BuildingsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

def compute_angle(edge):

    # compute angle
    y2, x2, y1, x1 = edge
    pc = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    pp = np.array([0, 1])
    pr = np.array([x1, y1]) if x1 >= x2 else np.array([x2, y2])
    pr -= pc
    cosine_angle = np.dot(pp, pr) / (np.linalg.norm(pp) * np.linalg.norm(pr))
    angle = np.arccos(cosine_angle)
    angle = 180.0 - np.degrees(angle)

    delta_degree = 10.0
    n_bins = 18
    bin_num = (int(angle/delta_degree+0.5)%n_bins)

    return bin_num

# Create model object.
INCLUDE_EDGE = False
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config, input_channel=3, corner_only=not INCLUDE_EDGE)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
#saved_model = './logs/la_dataset20190410T1507/mask_rcnn_la_dataset_0037.pth'
#saved_model = './logs/trainingdoubleset220190830T1803/mask_rcnn_trainingdoubleset2_0030.pth'
saved_model = './logs/outdoordataset20200306T1609/mask_rcnn_outdoordataset_0030.pth'

model.load_state_dict(torch.load(saved_model))

# _, last_saved = m odel.find_last()
# model.load_state_dict(torch.load(last_saved))
print('loaded weights from {}'.format(saved_model))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'edge', 'corner']

# Load a random image from the images folder
#im_path = '/local-scratch/fza49/cities_dataset/double_list2.txt'
im_path = '/local-scratch/fuyang/cities_dataset/valid_list.txt'
with open(im_path) as f:
    im_list = [x.strip()+'.jpg' for x in f.readlines()]
file_names = im_list

for fname in file_names:
    
    image = skimage.io.imread(os.path.join(IMAGE_DIR, fname))
    _id = fname.replace('.jpg', '')

    # np.set_printoptions(threshold=np.nan)

    # Run detection
    xs = np.tile((np.arange(0, 256)[np.newaxis, :]), (256, 1))
    ys = xs.transpose(1, 0)
    xs = xs[:, :, np.newaxis]
    ys = ys[:, :, np.newaxis]

    with torch.no_grad():
        results, outline = model.detect([image], None, None, None)

    outline_im = Image.fromarray(255.0*outline.reshape(256, 256))
    # plt.imshow(outline_im)
    # plt.show()

    # Visualize results
    r = results[0]

    # Grab corners and edges
    edges = r['rois'][r['class_ids'] == 1, :]
    corners = r['rois'][r['class_ids'] == 2, :]
    ce_probs = r['relations']
    edges_dir = r['edges_dir'][r['class_ids'] == 1, :]

    # Draw corners and edges in svg
    im_path = os.path.join(IMAGE_DIR, _id + '.jpg')
    dwg = svgwrite.Drawing(os.path.join(SAVE_DIR, 'svg', '{}_2.svg'.format(_id)), (256, 256))
    dwg.add(svgwrite.image.Image(im_path, size=(256, 256)))
    det_corners = utils.draw_detections(dwg, edges, corners, edges_dir, ce_probs, use_dist=False, corner_only=True, edge_boost=False)
    np.save(os.path.join(SAVE_DIR, 'npy', _id), det_corners)

    #utils.reconstruct(dwg, corners, ce_probs.reshape(*ce_probs.shape[1:]))
    #utils.build_overcomplex_graph(dwg, corners, edges, edges_dir, ce_probs.reshape(*ce_probs.shape[1:]))

    dwg.save()
