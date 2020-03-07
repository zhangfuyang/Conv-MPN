"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import math
import random
import cv2
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.color
import skimage.io
import torch
from collections import defaultdict
import pdb

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        # from PIL import Image, ImageDraw
        # import matplotlib.pyplot as plt
        # im = Image.fromarray(mask[:, :, i]).convert('RGB')
        # dr = ImageDraw.Draw(im)

        # max_size = max(y2-y1, x2-x1)
        # yc = (y2+y1)/2.0
        # xc = (x2+x1)/2.0
        gap = 1
        n_y1, n_x1 = max(y1-gap, 0), max(x1-gap, 0)
        n_y2, n_x2 = min(y2+gap, mask.shape[0]-1), min(x2+gap, mask.shape[0]-1)

        boxes[i] = np.array([n_y1, n_x1, n_y2, n_x2])
        # boxes[i] = np.array([y1, x1, y2, x2])

        # dr.rectangle(((x1-gap, y1-gap), (x2+gap, y2+gap)), outline="blue")
        # plt.imshow(im)
        # plt.show()
        
    return boxes.astype(np.int32)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result

def compute_distance(corners, edges):
    yc, xc, _, _ = corners.chunk(4, dim=-1)
    ye1, xe1, ye2, xe2 = edges.chunk(4, dim=-1)
    ce_dist = []
    for k in range(corners.shape[0]):
        dist1 = torch.sqrt((ye1-yc[k])**2 + (xe1-xc[k])**2)
        dist2 = torch.sqrt((ye2-yc[k])**2 + (xe2-xc[k])**2)

        # print(yc[k], xc[k])
        # for a, b, c, d in zip(ye1[:2], xe1[:2], ye2[:2], xe2[:2]):
        #     print('p1')
        #     print(a, b)
        #     print('p2')
        #     print(c, d)
        # print('dist')
        # print(dist1[:2])
        # print(dist2[:2])

        dist_comb = torch.cat([dist1, dist2], dim=-1)
        dist, _ = torch.min(dist_comb, dim=-1)
        ce_dist.append(dist)
    ce_dist = torch.stack(ce_dist, dim=0)
    return ce_dist
# def compute_corners_location(boxes):
#     y1, x1, y2, x2 = boxes.chunk(4, dim=-1)
#     yc = (y1+y2)/2.0
#     xc = (x1+x2)/2.0
#     return  torch.cat([yc, xc], dim=1)

# def compute_edges_endpoints(boxes, edges_dir):
#     y1, x1, y2, x2 = boxes.chunk(4, dim=-1)
#     height = y2-y1
#     width = x2 -x1
#     center_y = (y1+y2)/2.0
#     center_x = (x1+x2)/2.0
#     _, indices = torch.max(edges_dir, dim=-1)

#     angle = 30.0*np.argmax(e)

#     x_shift = np.clip((height/2.0)*np.tan(np.radians(angle)), -width/2.0, width/2.0)
#     y_shift = np.clip((width/2.0)*np.tan(np.radians(90.0-angle)), -height/2.0, height/2.0)

#     if angle < 90.0:
#         draw.line((xc-x_shift, yc+y_shift, xc+x_shift, yc-y_shift), fill='blue', width=2)
#     else:
#         draw.line((xc-x_shift, yc-y_shift, xc+x_shift, yc+y_shift), fill='blue', width=2)
#     return

############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, rot, flip, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
            "flip": flip,
            "rot": rot
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Evaluations
############################################################

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def edge_endpoints(e, d, delta_degree=10.0):
    y1, x1, y2, x2 = e
    height = (y2-y1)
    width = (x2-x1)
    xc, yc = (x1+x2)/2.0, (y1+y2)/2.0
    angle = delta_degree*np.argmax(d)
    x_shift = np.clip((height/2.0)*np.tan(np.radians(angle)), -width/2.0, width/2.0)
    y_shift = np.clip((width/2.0)*np.tan(np.radians(90.0-angle)), -height/2.0, height/2.0)

    if angle < 90.0:
        return np.array([yc+y_shift, xc-x_shift, yc-y_shift, xc+x_shift])
    else:
        return np.array([yc-y_shift, xc-x_shift, yc+y_shift, xc+x_shift])


def primitive_detector_eval(all_dets,
                            roidb,
                            vocabulary,
                            ovthresh=0.3,
                            use_07_metric=False,
                            edge_boost_corner=True):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the evaluation for detector.
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    # ground-truth bboxes for every class in each image
    npos_all = [0 for _ in range(len(vocabulary))]
    gt_records = [[{'bbox': list(), 'det': list(), 'mask': list(), 'edge_dir': list()} for _ in range(len(vocabulary))] for _ in
                  range(len(roidb))]
    for image_id, roi_info in enumerate(roidb):
        gt_classes = roi_info['gt_classes'].tolist()
        gt_bboxes = roi_info['boxes'].tolist()
        gt_masks = roi_info['masks'].tolist()
        gt_edge_dirs = roi_info['gt_edge_dirs'].tolist()
        assert (len(gt_classes) == len(gt_bboxes) == len(gt_masks))
        for gt_class, gt_bbox, gt_mask, gt_edge_dir in zip(gt_classes, gt_bboxes, gt_masks, gt_edge_dirs):
            gt_class = int(gt_class)
            gt_records[image_id][gt_class]['bbox'].append(gt_bbox)
            gt_records[image_id][gt_class]['det'].append(False)
            gt_records[image_id][gt_class]['mask'].append(gt_mask)
            gt_records[image_id][gt_class]['edge_dir'].append(gt_edge_dir)
            npos_all[gt_class] = npos_all[gt_class] + 1

    # one entry for each class
    image_ids_all = [[] for _ in range(len(vocabulary))]
    confidence_all = [[] for _ in range(len(vocabulary))]
    dets_all = [[] for _ in range(len(vocabulary))]
    edge_dirs_all = [[] for _ in range(len(vocabulary))]

    for class_idx in range(len(vocabulary)):
        for image_id in range(len(all_dets[class_idx])):
            # iterate through each det
            for det in all_dets[class_idx][image_id]:
                image_ids_all[class_idx].append(image_id)
                confidence_all[class_idx].append(det[4])
                dets_all[class_idx].append(list(det[:4]))
                if class_idx == 1:
                    edge_dirs_all[class_idx].append(list(det[5:]))

    # compute map for every selected class

    if edge_boost_corner:
        corner_dets_from_edge = list()
        corner_confs_from_edge = list()
        corner_image_ids_from_edge = list()

    for class_idx in range(1, len(vocabulary)):
        npos = npos_all[class_idx]

        image_ids = image_ids_all[class_idx]
        confidence = np.array(confidence_all[class_idx])
        dets = np.array(dets_all[class_idx])
        edge_dirs = np.array(edge_dirs_all[class_idx])

        nd = len(image_ids)

        if class_idx == 2:  # evaluation for corners
            if edge_boost_corner:
                corner_dets_from_edge = np.array(corner_dets_from_edge)
                corner_confs_from_edge = np.array(corner_confs_from_edge)
                corner_image_ids_from_edge = np.array(corner_image_ids_from_edge)
                dets = np.concatenate([dets, corner_dets_from_edge], axis=0)
                confidence = np.concatenate([confidence, corner_confs_from_edge])
                image_ids = np.concatenate([image_ids, corner_image_ids_from_edge])

                dets, confidence, image_ids = _corner_nms(dets, confidence, image_ids, threshold=3)

                nd = len(image_ids)

            tp = np.zeros(nd)
            fp = np.zeros(nd)

            if dets.shape[0] > 0:
                # sort by confidence
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                dets = dets[sorted_ind, :]
                image_ids = [image_ids[x] for x in sorted_ind]

                # go down dets and mark TPs and FPs
                for d in range(nd):
                    R = gt_records[image_ids[d]]
                    det = dets[d, :].astype(float)
                    ovmax = -np.inf
                    BBGT = np.array(R[class_idx]['bbox']).astype(float)

                    if BBGT.size > 0:
                        # compute overlaps
                        # intersection
                        # the format is y1, x1, y2, x2
                        bb_corner = [det[0]-4, det[1]-4, det[0]+4, det[1]+4]
                        ixmin = np.maximum(BBGT[:, 1], bb_corner[1])
                        iymin = np.maximum(BBGT[:, 0], bb_corner[0])
                        ixmax = np.minimum(BBGT[:, 3], bb_corner[3])
                        iymax = np.minimum(BBGT[:, 2], bb_corner[2])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        # union
                        uni = ((bb_corner[2] - bb_corner[0] + 1.) * (bb_corner[3] - bb_corner[1] + 1.) +
                               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)

                    if ovmax > ovthresh:
                        if not R[class_idx]['det'][jmax]:
                            tp[d] = 1.
                            R[class_idx]['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric)

            print('rec: ', rec)
            print('prec: ', prec)
            print('ap: ', ap)
            print('total positive examples for corner is {}, detected {} of them'.format(npos, tp[-1]))

        elif class_idx == 1:  # for edges
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            mask_ious = list()

            if dets.shape[0] > 0:
                # sort by confidence
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                dets = dets[sorted_ind, :]
                edge_dirs = edge_dirs[sorted_ind, :]
                image_ids = [image_ids[x] for x in sorted_ind]

                # go down dets and mark TPs and FPs
                for d in range(nd):
                    R = gt_records[image_ids[d]]
                    det = dets[d, :].astype(float)
                    edge_dir = edge_dirs[d, :].astype(float)

                    ovmax = -np.inf
                    BBGT = np.array(R[class_idx]['bbox']).astype(float)
                    direction_gt = np.array(R[class_idx]['edge_dir']).astype(float)

                    if edge_boost_corner:
                        # here we boost the corner detection results with our edge detections
                        y1, x1, y2, x2 = edge_endpoints(det, edge_dir)
                        corner_dets_from_edge.append([y1, x1, -1, -1])
                        corner_confs_from_edge.append(confidence[d])
                        corner_image_ids_from_edge.append(image_ids[d])
                        corner_dets_from_edge.append([y2, x2, -1, -1])
                        corner_confs_from_edge.append(confidence[d])
                        corner_image_ids_from_edge.append(image_ids[d])


                    if BBGT.size > 0:
                        # compute overlaps
                        # intersection
                        # the format is y1, x1, y2, x2
                        bb_edge = det

                        ixmin = np.maximum(BBGT[:, 1], bb_edge[1])
                        iymin = np.maximum(BBGT[:, 0], bb_edge[0])
                        ixmax = np.minimum(BBGT[:, 3], bb_edge[3])
                        iymax = np.minimum(BBGT[:, 2], bb_edge[2])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        # union
                        uni = ((bb_edge[2] - bb_edge[0] + 1.) * (bb_edge[3] - bb_edge[1] + 1.) +
                               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)

                    if ovmax > ovthresh:
                        if not R[class_idx]['det'][jmax]:
                            if R[class_idx]['edge_dir'][jmax] == np.argmax(edge_dir):
                                tp[d] = 1.
                                R[class_idx]['det'][jmax] = 1
                            else:
                                fp[d] = 1.
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.

                # compute precision recall
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / float(npos)
                # avoid divide by zero in case the first detection matches a difficult
                # ground truth
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                ap = voc_ap(rec, prec, use_07_metric)

                print('rec: ', rec)
                print('prec: ', prec)
                print('ap: ', ap)
                print('total positive examples for edge is {}, detected {} of them'.format(npos, tp[-1]))
                # mean_mask_iou = np.array(mask_ious).mean()
                # print('averge iou for all positive detections: {}'.format(mean_mask_iou))


def _corner_nms(corners, confidences, image_ids, threshold=4):
    corners_by_image = defaultdict(list)
    nd = len(image_ids)

    for d in range(nd):
        image_id = image_ids[d]
        corners_by_image[image_id].append((corners[d], d))

    selected_indices_all = list()
    # Run NMS on every image
    for image_id in corners_by_image.keys():
        selected_corners = list()
        selected_indices = list()
        for corner, idx in corners_by_image[image_id]:
            insert_flag = True
            for other in selected_corners:
                dist = np.linalg.norm(corner[:2] - other[:2])
                if dist < threshold:
                    insert_flag = False
                    break
            if insert_flag:
                selected_indices.append(idx)
                selected_corners.append(corner)
        selected_indices_all += selected_indices

    return corners[selected_indices_all], confidences[selected_indices_all], image_ids[selected_indices_all]




def compute_mask_iou(gt_mask, mask_pred, bbox_pred, MAX_SIZE=255):
    bbox_pred = [min(MAX_SIZE, coord) for coord in bbox_pred]
    cropped_pred = mask_pred[int(bbox_pred[0]):int(bbox_pred[2])+1, int(bbox_pred[1]):int(bbox_pred[3])+1].astype(bool)
    resized_gt = cv2.resize(gt_mask, (int(bbox_pred[3]) - int(bbox_pred[1]) + 1, int(bbox_pred[2]) - int(bbox_pred[0]) + 1))
    resized_gt = resized_gt.round().astype(bool)
    inter = (resized_gt & cropped_pred).sum()

    union = (resized_gt | cropped_pred).sum()
    iou = inter / union

    return iou

def reconstruct(dwg, corners, relations):

    # format
    relations = relations.reshape(relations.shape[0], -1)
    relations = relations.transpose(1, 0)

    # get top 2
    ind = np.argsort(relations, axis=-1)
    ind = ind[:, -2:]
    print(ind.shape)
    for i, j in ind:
        c1 = corners[i, :]
        c2 = corners[j, :]
        y1, x1 = (c1[0]+c1[2])/2.0, (c1[1]+c1[3])/2.0
        y2, x2 = (c2[0]+c2[2])/2.0, (c2[1]+c2[3])/2.0
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='blue', stroke_width=3, opacity=1))

    for i, j in ind:
        c1 = corners[i, :]
        c2 = corners[j, :]
        y1, x1 = (c1[0]+c1[2])/2.0, (c1[1]+c1[3])/2.0
        y2, x2 = (c2[0]+c2[2])/2.0, (c2[1]+c2[3])/2.0
        dwg.add(dwg.circle(center=(x1, y1),r=3, stroke='red', fill='white', stroke_width=1, opacity=1))
        dwg.add(dwg.circle(center=(x2, y2),r=3, stroke='red', fill='white', stroke_width=1, opacity=1))
    return

def build_overcomplex_graph(dwg, corners, edges, edges_dir, relations):

    # format
    thres = .5
    Nc = corners.shape[0]
    Ne = edges.shape[0]
    new_relations = []
    new_edges = []
    new_corners = []


    # get relations corner->edge
    ind = np.argsort(relations, axis=-1)
    val = np.sort(relations, axis=-1)
    for i in range(Nc):
        num = np.sum(val[i, :]>thres)
        if num >=2:
            es = list(ind[i, np.where(val[i, :]>thres)].ravel())
        else:
            es = list(ind[i, -2:])
        for j in es:
            new_relations.append((i, j))

        new_edges += es
        new_corners.append(i)

    new_edges = list(set(new_edges))
    print(new_edges)

    # get relations edge->corner
    relations = relations.transpose(1, 0)
    ind = np.argsort(relations, axis=-1)
    val = np.sort(relations, axis=-1)
    
    for i in range(Ne):
        num = np.sum(val[i, :]>thres)
        if num >=2:
            cs = list(ind[i, np.where(val[i, :]>thres)].ravel())
        else:
            cs = list(ind[i, -2:])
        for j in cs:    
            new_relations.append((j, i))
        new_corners += cs
        new_edges.append(i)

    new_corners = list(set(new_corners))
    new_relations = list(set(new_relations))

    # draw overcomplex graph
    for i in new_edges:
        e = edges[i]
        d = edges_dir[i]
        y1, x1, y2, x2 = edge_endpoints(e, d)
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='blue', stroke_width=3, opacity=.5))

    for i in new_corners:
        c = corners[i]
        y, x = (c[0]+c[2])/2.0, (c[1]+c[3])/2.0
        dwg.add(dwg.circle(center=(x, y),r=3, stroke='green', fill='white', stroke_width=1, opacity=1))


    for i, j in new_relations:

        c = corners[i]
        y, x = (c[0]+c[2])/2.0, (c[1]+c[3])/2.0

        e = edges[j]
        d = edges_dir[j]
        y1, x1, y2, x2 = edge_endpoints(e, d)
        center_x = (x1+x2)/2.0
        center_y = (y1+y2)/2.0
        dwg.add(dwg.line((float(x), float(y)), (float(center_x), float(center_y)), stroke='magenta', stroke_width=1, opacity=.8))

    return

def draw_relations(dwg, corners, edges, corners_conf, edges_conf, left_relations, right_relations, use_dist=False, dist_thres=8.0):
    tresh = .5

    # Draw edges
    for i, pts in (zip(range(edges.shape[0]), edges)):
        if edges_conf[i] > tresh:
            y1, x1, y2, x2 = pts
            dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='green', stroke_width=3, opacity=.8))
    
    # Draw corners
    for i in range(corners.shape[0]):
        y, x = corners[i, :2]
        y, x = float(y), float(x)
        if corners_conf[i] > tresh:
            dwg.add(dwg.circle(center=(x,y),r=2, stroke='green', fill='white', stroke_width=1, opacity=.8))
    
    # # Draw relations
    # if use_dist == False:
    #     for i in range(edges.shape[0]):
    #         y1, x1, y2, x2 = edges[i, :]
    #         if edges_conf[i] > tresh:
    #             lidx = np.argmax(left_relations[0, :, i])
    #             lpt = corners[lidx, :2]
    #             lconf = left_relations[0, lidx, i]

    #             ridx = np.argmax(right_relations[0, :, i])
    #             rpt = corners[ridx, :2]
    #             rconf = right_relations[0, ridx, i]

    #             if corners_conf[lidx] > tresh and lconf > 0:
    #                 dwg.add(dwg.line((float(lpt[1]), float(lpt[0])), (float(x1+x2)/2.0, float(y1+y2)/2.0), stroke='magenta', stroke_width=1, opacity=.7))
    #             if corners_conf[ridx] > tresh and rconf > 0:
    #                 dwg.add(dwg.line((float(rpt[1]), float(rpt[0])), (float(x1+x2)/2.0, float(y1+y2)/2.0), stroke='magenta', stroke_width=1, opacity=.7))
    # else:
    #     for i in range(edges.shape[0]):
    #         y1, x1, y2, x2 = edges[i, :]
    #         if edges_conf[i] > tresh:
    #             lidx = np.argmin(left_relations[:, i])
    #             lpt = corners[lidx, :2]
    #             lconf = left_relations[lidx, i]

    #             ridx = np.argmin(right_relations[:, i])
    #             rpt = corners[ridx, :2]
    #             rconf = right_relations[ridx, i]

    #             if corners_conf[lidx] > tresh and (0 < lconf < dist_thres):
    #                 dwg.add(dwg.line((float(lpt[1]), float(lpt[0])), (float(x1+x2)/2.0, float(y1+y2)/2.0), stroke='magenta', stroke_width=1, opacity=.7))
    #             if corners_conf[ridx] > tresh and (0 < rconf < dist_thres):
    #                 dwg.add(dwg.line((float(rpt[1]), float(rpt[0])), (float(x1+x2)/2.0, float(y1+y2)/2.0), stroke='magenta', stroke_width=1, opacity=.7))

def compute_dists(corners_det, edges_det, thresh=8.0):

    # compute corner dist
    y, x, _, _ = np.split(corners_det, 4, axis=-1)
    c_dist = np.sqrt((x - x.transpose(1, 0))**2 + (y - y.transpose(1, 0))**2)
    ind_pos = c_dist<thresh
    ind_neg = c_dist>=thresh
    c_dist[ind_pos] = 1.0
    c_dist[ind_neg] = 0.0
    np.fill_diagonal(c_dist, 0.0)

    # compute edge dist
    y1, x1, y2, x2 = np.split(edges_det, 4, axis=-1)
    y3, x3, y4, x4 = np.split(edges_det, 4, axis=-1)

    dist13 = np.sqrt((x1 - x3.transpose(1, 0))**2 + (y1 - y3.transpose(1, 0))**2)
    dist14 = np.sqrt((x1 - x4.transpose(1, 0))**2 + (y1 - y4.transpose(1, 0))**2)
    dist23 = np.sqrt((x2 - x3.transpose(1, 0))**2 + (y2 - y3.transpose(1, 0))**2)
    dist24 = np.sqrt((x2 - x4.transpose(1, 0))**2 + (y2 - y4.transpose(1, 0))**2)

    d1 = dist13 + dist24
    d2 = dist14 + dist23

    e_dist = np.stack([d1, d2], axis=-1)
    e_dist = np.min(e_dist, axis=-1)
    ind_pos = e_dist<thresh*2
    ind_neg = e_dist>=thresh*2
    e_dist[ind_pos] = 1.0
    e_dist[ind_neg] = 0.0
    np.fill_diagonal(e_dist, 0.0)

    # compute corner-edge dist
    dist1 = np.sqrt((x - x1.transpose(1, 0))**2 + (y - y1.transpose(1, 0))**2)
    dist2 = np.sqrt((x - x2.transpose(1, 0))**2 + (y - y2.transpose(1, 0))**2)
    r_dist = np.stack([dist1, dist2], axis=-1)
    r_dist = np.min(r_dist, axis=-1)

    ind_pos = r_dist<thresh
    ind_neg = r_dist>=thresh
    raw_dist = np.array(r_dist)
    r_dist[ind_pos] = 1.0
    r_dist[ind_neg] = 0.0

    return c_dist, e_dist, r_dist, raw_dist

def split_dist_mat(mat, v1, v2):
    y1, x1, _, _ = np.split(v1, 4, axis=-1)
    y2, x2, y3, x3 = np.split(v2, 4, axis=-1)
    yc, xc = (y3+y2)/2.0, (x3+x2)/2.0
    dy, dx = y3-y2, x3-x2
    ny, nx = y3-y2, x2-x3

    # ind = (dx == 0)
    # print(ny[ind])
    # print(nx[ind])
    # ny[ind] = 1.0
    # ind = (dy == 0)
    # nx[ind] = 1.0

    ny = np.tile(ny, (1, x1.shape[0])).transpose(1, 0)
    nx = np.tile(nx, (1, x1.shape[0])).transpose(1, 0)
    x1mc = x1-xc.transpose(1, 0)
    y1mc = y1-yc.transpose(1, 0)
    d = y1mc*nx - x1mc*ny

    # handle vertical/horizontal edges
    yc = np.tile(yc, (1, x1.shape[0])).transpose(1, 0)
    xc = np.tile(xc, (1, x1.shape[0])).transpose(1, 0)
    y1 = np.tile(y1, (1, x2.shape[0]))
    x1 = np.tile(x1, (1, x2.shape[0]))
    dx = x1-xc
    dy = y1-yc
    pos = (nx == 0)&(dy >= 0)
    neg = (nx == 0)&(dy < 0)
    d[pos] = 1.0
    d[neg] = -1.0
    pos = (ny == 0)&(dx >= 0)
    neg = (ny == 0)&(dx < 0)
    d[pos] = 1.0
    d[neg] = -1.0

    # ydel, xdel = y1-yc.transpose(1, 0), x1-xc.transpose(1, 0)
    # lcond = ((xdel<=0)&(ydel>0))|((xdel<0)&(ydel<=0))
    # rcond = ((xdel>0)&(ydel>0))|((xdel<=0)&(ydel<=0))

    # lmat = lcond * mat
    # rmat = rcond * mat
    lmat = (d >= 0) * mat
    rmat = (d < 0) * mat

    return lmat, rmat


def draw_detections(dwg, edges, corners, edges_dir, ce_probs, use_dist=False, corner_only=False, edge_boost=False):

    # Draw edges
    edges_coords = []
    for j, e, d in zip(range(edges.shape[0]), edges, edges_dir):
        y1, x1, y2, x2 = edge_endpoints(e, d)
        edges_coords.append([y1, x1, y2, x2])

    # Draw corners
    corners_coords = []
    for i, c in enumerate(corners):
        y1, x1, y2, x2 = c
        yc = (y2+y1)/2.0
        xc = (x2+x1)/2.0
        corners_coords.append([yc, xc, -1, -1])

    # setup
    corners_det = np.array(corners_coords)
    edges_det = np.array(edges_coords)
    edges_conf = np.ones(edges_det.shape[0])
    corners_conf = np.ones(corners_det.shape[0])

    if corner_only:
        # Draw corners only
        tresh = 0.5

        if edge_boost:
            new_corner_dets = list()
            for i in range(edges_det.shape[0]):
                edge_coords = edges_coords[i]
                new_corner_dets.append(edge_coords[:2])
                new_corner_dets.append(edge_coords[2:])
            new_corner_dets = np.array(new_corner_dets)
            corners_det = np.concatenate([corners_det[:, :2], new_corner_dets], axis=0)
        else:
            corners_det = corners_det[:, :2]

        selected_corners_det = list()

        for corner in corners_det:
            insert_flag = True
            for other in selected_corners_det:
                if np.linalg.norm(corner - other) < 3:
                    insert_flag = False
                    break
            if insert_flag:
                selected_corners_det.append(corner)

        for i in range(len(selected_corners_det)):
            y, x = selected_corners_det[i]
            y, x = float(y), float(x)
            dwg.add(dwg.circle(center=(x, y), r=2, stroke='green', fill='white', stroke_width=1, opacity=.8))

        return selected_corners_det





    # compute dists
    c_dist, e_dist, r_dist, raw_dist = compute_dists(corners_det, edges_det)
    lmat, rmat = split_dist_mat(r_dist, corners_det, edges_det)
    left_relations = np.array(ce_probs)*np.array(lmat)*np.array(r_dist)
    right_relations = np.array(ce_probs)*np.array(rmat)*np.array(r_dist)
    left_dist = np.array(raw_dist)*np.array(lmat)
    left_dist[lmat==0] = 999999
    right_dist = np.array(raw_dist)*np.array(rmat)  
    right_dist[rmat==0] = 999999

    # draw relations
    # draw_relations(dwg, corners_det, edges_det, corners_conf, edges_conf, left_dist, right_dist, use_dist=use_dist)
    draw_relations(dwg, corners_det, edges_det, corners_conf, edges_conf, left_relations, right_relations, use_dist=use_dist)


def draw_gt(dwg, im_path):

    # load annotations
    p_path = im_path.replace('.jpg', '.npy').replace('rgb', 'annots')
    v_set = np.load(open(p_path, 'rb'),  encoding='bytes')
    graph = dict(v_set[()])

    # draw groundtruth
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
            dwg.add(dwg.line((2*float(x1), 2*float(y1)), (2*float(x2), 2*float(y2)), stroke='black', stroke_width=1, opacity=.8))
    return

def rotate_coords(image_shape, xy, angle):
    org_center = (image_shape-1)/2.
    rot_center = (image_shape-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center

def generate_boxes_from_gt(im_path, rot, flip):

    # load annotations
    p_path = im_path.replace('.jpg', '.npy').replace('rgb', 'annot')
    v_set = np.load(open(p_path, 'rb'),  encoding='bytes', allow_pickle=True)
    graph = dict(v_set[()])

    # draw mask
    class_ids, coords = [], []

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

    edge_list = list(edge_set)
    for edge in edge_list:

        # apply augmentation
        x1, y1, x2, y2 = edge
        x1, y1 = rotate_coords(np.array([256, 256]), np.array([x1, y1]), rot)
        x2, y2 = rotate_coords(np.array([256, 256]), np.array([x2, y2]), rot)

        if flip:
            x1, y1 = (128-abs(128-x1), y1) if x1 > 128 else (128+abs(128-x1), y1)
            x2, y2 = (128-abs(128-x2), y2) if x2 > 128 else (128+abs(128-x2), y2)
        class_ids.append(1)
        coords.append([y1, x1, y2, x2])

    # prepare corner instances for this image
    for v in graph:

        # apply augmentation during training
        x, y = v
        x, y = rotate_coords(np.array([256, 256]), np.array([x, y]), rot)
        if flip:
            x, y = (128-abs(128-x), y) if x > 128 else (128+abs(128-x), y)
        class_ids.append(2)
        coords.append([y, x, -1, -1])

    class_ids = np.array(class_ids).astype('int32')
    coords = np.array(coords)

    return class_ids, coords
