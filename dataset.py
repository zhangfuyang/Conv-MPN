import numpy as np
import random
from torch.utils.data import Dataset
import os
import skimage
import cv2
from config import *


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
class Graphdataset(Dataset):
    def __init__(self, datapath,
                 detcornerpath, phase='train',
                 full_connected_init=True, mix_gt=False, demo=False,
                 per_edge=False):
        super(Graphdataset, self).__init__()
        self.datapath = datapath
        self.per_edge_classifier = per_edge
        self.detcornerpath = detcornerpath
        self.phase = phase
        self.mix_gt = mix_gt
        self.full_connected_init = full_connected_init
        self.demo = demo
        if phase == 'train':
            datalistfile = os.path.join(datapath, 'train_list.txt')
        else:
            datalistfile = os.path.join(datapath, 'valid_list.txt')
        with open(datalistfile, 'r') as f:
            self._data_names = f.readlines()

    def __len__(self):
        return len(self._data_names)

    def get_annot(self, data_name):
        annot = np.load(os.path.join(self.datapath, 'annot', data_name+'.npy'),
                        allow_pickle=True, encoding='latin1').tolist()
        return annot
    
    def getbyname(self, name):
        for i in range(len(self._data_names)):
            if self._data_names[i][:-1] == name:
                return self.__getitem__(i)

    def __getitem__(self, idx):
        data_name = self._data_names[idx][:-1]
        rgb = skimage.img_as_float(cv2.imread(os.path.join(self.datapath, 'rgb', data_name+'.jpg')))
        annot = np.load(os.path.join(self.datapath, 'annot', data_name+'.npy'),
                        allow_pickle=True, encoding='latin1').tolist()
        corners = np.array(np.load(
            os.path.join(self.detcornerpath, data_name + '.npy'), allow_pickle=True))  # [N, 2]


        if self.per_edge_classifier:
            img = rgb.transpose((2,0,1))
            img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
            loc1, loc2 = random.sample(list(annot.keys()), 0)

            flag = True
            for neighbor in annot[loc1]:
                if (loc2[0] - neighbor[0]) ** 2 + (loc2[1] - neighbor[1]) ** 2 < 1:
                    label = 1
                    flag = False
                    break
            if flag:
                label = 0
            loc1 = list(loc1)
            loc2 = list(loc2)
            loc1[0] -= np.random.normal(0, 0)
            loc1[1] -= np.random.normal(0, 0)
            loc2[0] -= np.random.normal(0, 0)
            loc2[1] -= np.random.normal(0, 0)
            corner_edge_map = np.zeros((256, 256)).astype(np.double)
            cv2.line(corner_edge_map, (int(loc1[0]), int(loc1[1])), (int(loc2[0]), int(loc2[1])), (1.0), 3)
            return {
                "x": corner_edge_map,
                "y": label,
                "img": img
            }
        # full_connect_init
        if self.mix_gt:
            corners = np.array(list(annot.keys()))[:, [1,0]]
            corners += np.random.normal(0, 0, size=corners.shape)
        cornerList = []
        for corner_i in range(corners.shape[0]):
            cornerList.append((corners[corner_i][1], corners[corner_i][0]))
        edge_masks = []
        edges = []
        for edge_i in range(corners.shape[0]):
            for edge_j in range(edge_i + 1, corners.shape[0]):
                edge_mask = np.zeros((256, 256)).astype(np.double)
                loc1 = np.array(cornerList[edge_i]).astype(np.int)
                loc2 = np.array(cornerList[edge_j]).astype(np.int)
                cv2.line(edge_mask, (loc1[0], loc1[1]), (loc2[0], loc2[1]), 1.0, 3)
                edge_masks.append(edge_mask)

                edges.append([edge_i, edge_j])
                edges.append([edge_j, edge_i])


        edges = np.array(edges).T  # [2, N * N - 1]

        gt_edge = self.get_gt(corners, annot)  # shape: [2, gt_edge_num]

        raw_data = {
            'name': data_name,
            'rgb': rgb,
            'edges': edges,
            'edge_feature': edge_masks,
            'corners': corners,
            'corners_feature': None,
            'gt': gt_edge,
            'annot': annot
        }
        return self.get_data(raw_data)

    def get_gt(self, preds, annot):
        """
        :param preds: preds(x,y) == annot(y,x)
        :param annot:
        :return:
        """
        gt_edges = set()
        gt_corners = list(annot.keys())
        if self.mix_gt:
            for corner_i in range(len(gt_corners)):
                for corner_neighbor in annot[gt_corners[corner_i]]:
                    for corner_j in range(len(gt_corners)):
                        if (gt_corners[corner_j][0] - corner_neighbor[0]) ** 2 + \
                            (gt_corners[corner_j][1] - corner_neighbor[1]) ** 2 < 1:
                            gt_edges.add((corner_i, corner_j))
                            gt_edges.add((corner_j, corner_i))
                            break
            return list(gt_edges)
        gt_map = {}
        match_id_set = set()
        for gt_corner_ in gt_corners:
            dist = 7
            match_idx = -1
            for pred_i in range(preds.shape[0]):
                if pred_i in match_id_set:
                    continue
                pred = preds[pred_i]
                temp_dist = np.sqrt((pred[0] - gt_corner_[1]) ** 2 + (pred[1] - gt_corner_[0]) ** 2)
                if temp_dist < dist:
                    dist = temp_dist
                    match_idx = pred_i
            match_id_set.add(match_idx)
            gt_map[gt_corner_] = match_idx
        if new_label:
            for gt_corner_ in gt_corners:
                dist = 15
                match_idx = -1
                if gt_map[gt_corner_] == -1:
                    for pred_i in range(preds.shape[0]):
                        pred = preds[pred_i]
                        temp_dist = np.sqrt((pred[0] - gt_corner_[1]) ** 2 + (pred[1] - gt_corner_[0]) ** 2)
                        if temp_dist < dist:
                            dist = temp_dist
                            match_idx = pred_i
                    gt_map[gt_corner_] = match_idx

            for gt_corner_ in gt_corners:
                if gt_map[gt_corner_] == -1:
                    continue
                for neighbor in annot[gt_corner_]:
                    if gt_map[tuple(neighbor)] == -1:
                        target_dir = (neighbor - np.array(gt_corner_)) / np.sqrt(np.sum((neighbor - np.array(gt_corner_)) ** 2))
                        # get neighbor's neighbor with same direction
                        cos_value = 0.5
                        neighbor_good = None
                        for neighbor_v2 in annot[tuple(neighbor)]:
                            if gt_map[tuple(neighbor_v2)] == -1:
                                continue
                            test_dir = (neighbor_v2 - neighbor) / np.sqrt(np.sum((neighbor_v2 - neighbor) ** 2))
                            if np.sum(test_dir * target_dir) > cos_value:
                                cos_value = np.sum(test_dir * target_dir)
                                neighbor_good = neighbor_v2
                        if neighbor_good is not None:
                            gt_edges.add((gt_map[gt_corner_], gt_map[tuple(neighbor_good)]))
                            gt_edges.add((gt_map[tuple(neighbor_good)], gt_map[gt_corner_]))
                        #else:
                        #    # we only looke twice
                        #    cos_value = 0.7
                        #    for neighbor_v2 in annot[tuple(neighbor)]:
                        #        for neighbor_v3 in annot[tuple(neighbor_v2)]:
                        #            if gt_map[tuple(neighbor_v3)] == -1:
                        #                continue
                        #            test_dir = (neighbor_v3 - neighbor) / np.sqrt(np.sum((neighbor_v3 - neighbor) ** 2))
                        #            if np.sum(test_dir * target_dir) > cos_value:
                        #                cos_value = np.sum(test_dir * target_dir)
                        #                neighbor_good = neighbor_v3
                        #    if neighbor_good is not None:
                        #        gt_edges.add((gt_map[gt_corner_], gt_map[tuple(neighbor_good)]))
                        #        gt_edges.add((gt_map[tuple(neighbor_good)], gt_map[gt_corner_]))

                    elif gt_map[tuple(neighbor)] == gt_map[gt_corner_]:
                        continue
                    else:
                        gt_edges.add((gt_map[gt_corner_], gt_map[tuple(neighbor)]))
                        gt_edges.add((gt_map[tuple(neighbor)], gt_map[gt_corner_]))
            return list(gt_edges)

        for gt_corner_ in gt_corners:
            if gt_map[gt_corner_] == -1:
                continue
            for neighbor in annot[gt_corner_]:
                if gt_map[tuple(neighbor)] == -1:
                    continue
                if gt_map[tuple(neighbor)] == gt_map[gt_corner_]:
                    continue
                gt_edges.add((gt_map[gt_corner_], gt_map[tuple(neighbor)]))
                gt_edges.add((gt_map[tuple(neighbor)], gt_map[gt_corner_]))
        return list(gt_edges)

    def get_data(self, data):
        img = data['rgb']
        corners = data['corners']
        edge_masks = data['edge_feature']
        gt = data['gt']
        annot = data['annot']

        edges = []
        for edge_i in range(corners.shape[0]):
            for edge_j in range(edge_i + 1, corners.shape[0]):
                edges.append((edge_i, edge_j))

        edge_index = []
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if edges[j][0] == edges[i][0] or edges[j][0] == edges[i][1] or \
                                edges[j][1] == edges[i][0] or edges[j][1] == edges[i][1]:
                    edge_index.append((i, j))
                    edge_index.append((j, i))
        edge_index = np.array(edge_index).T

        y = []
        for corner_i in range(corners.shape[0]):
            for corner_j in range(corner_i + 1, corners.shape[0]):
                if (corner_i, corner_j) in gt or (corner_j, corner_i) in gt:
                    y.append(1)
                else:
                    y.append(0)
        y = torch.Tensor(y).long()

        # process feature map for corners
        x = torch.Tensor(edge_masks).double()

        edge_index = torch.Tensor(edge_index).long()
        img = img.transpose((2,0,1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]

        if self.per_edge_classifier:
            choice_id = random.randint(0, y.shape[0] - 1)
            return {
                "x": x[choice_id],
                "y": y[choice_id],
                "img": img,
                "pos": corners,
                "annot": annot,
                "name": data['name']
            }

        return {
            "x": x,
            "edge_index": edge_index,
            "y": y,
            "img": img,
            "pos": corners,
            "annot": annot,
            "name": data['name']
        }

    def get_neighbor(self, corner_idx, edge_index):
        neighbor_ids = set()
        for j in range(edge_index.shape[1]):
            if corner_idx == edge_index[0, j]:
                neighbor_ids.add(edge_index[1, j])
            if corner_idx == edge_index[1, j]:
                neighbor_ids.add(edge_index[0, j])

        return list(neighbor_ids)
