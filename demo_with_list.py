import matplotlib.pyplot as plt
from dataset import Graphdataset
import numpy as np
from config import *
from utils import ensure_folder
from SVG_utils import svg_generate
from model import graphNetwork
from metrics import Metrics
import skimage
import cv2
import os
import torch
from drn import drn_c_26

no_svg = False
no_render = True
no_metric = False
save_npy = True
restore_npy = False
#threshold = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
threshold = [0.5]
save_folder = "conv_mpn_loop_3_pretrain_2"
checkpoint_name = 'checkpoint_14_0.916'
model_loop_time = 3
conv_mpn = True
gnn = False
#
#save_folder = "conv_mpn_loop_2_new"
#checkpoint_name = 'checkpoint_62_0.712'
#
#save_folder = "conv_mpn_loop_1"
#checkpoint_name = 'checkpoint_16_2.025'
#
#save_folder = "per_edge_classification_new"
#checkpoint_name = 'checkpoint_57_0.224'

def main():
    checkpoint = '{}/{}.tar'.format(save_folder, checkpoint_name)  # model checkpoint
    print('checkpoint: ' + str(checkpoint))
    # Load model
    checkpoint = torch.load(checkpoint, map_location=device)
    param = checkpoint['model'].state_dict()
    drn = drn_c_26(pretrained=False, image_channels=4)
    drn = torch.nn.Sequential(*list(drn.children())[:-7])
    model = graphNetwork(model_loop_time, backbone=drn ,conv_mpn=conv_mpn, gnn=gnn)
    model.double()
    model.load_state_dict(param, strict=True)
    model = model.to(device)
    print(model)
    model.eval()
    metrics = Metrics()
    metrics.reset()

    DATAPATH='/local-scratch/fza49/new/to_send'
    DETCORNERPATH='/local-scratch/fza49/new/to_send/corners'
    listfile = '/local-scratch/fza49/new/to_send/valid.txt'
    with open(listfile, 'r') as f:
        _data_names = f.readlines()
    for data_name in _data_names:
        rgb = skimage.img_as_float(cv2.imread(os.path.join(DATAPATH, 'rgb', data_name[:-1]+'.jpg')))
        corners = np.array(np.load(
            os.path.join(DETCORNERPATH, data_name[:-1] + '.npy'), allow_pickle=True))  # [N, 2]

        img = rgb.transpose((2,0,1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        
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

        if restore_npy:
            preds = np.load(save_folder + '/' + checkpoint_name + '/npy/' + data['name'] + '.npy')
            feature_map_vis = {}
        else:
            img = torch.tensor(img).to(device)
            edge_index = torch.tensor(edge_index).long().to(device)
            edge_masks = torch.tensor(edge_masks).double().to(device)
            with torch.no_grad():
                preds = model(img, edge_masks, edge_index)
            preds = preds.cpu().numpy()
        prob = np.exp(preds) / np.sum(np.exp(preds), 1)[..., np.newaxis]
        if save_npy:
            junc = corners[:, [1,0]]
            lines_on = []
            juncs_on = []
            _count = 0
            for corner_i in range(junc.shape[0]):
                for corner_j in range(corner_i + 1, junc.shape[0]):
                    if prob[_count, 1] > 0.8:
                    #if preds[_count].argmax() == 1:
                        lines_on.append((corner_i, corner_j))
                        if corner_i not in juncs_on:
                            juncs_on.append(corner_i)
                        if corner_j not in juncs_on:
                            juncs_on.append(corner_j)
                    _count += 1
            data_save = {
                'junctions': junc,
                'juncs_on': juncs_on,
                'lines_on': lines_on
            }
            np.save(DATAPATH + '/npy_8/' + data_name[:-1], data_save)
        if no_svg is False:
            colors = []
            nodes = corners.astype(np.int)
            nodes = nodes[:, [1, 0]]
            edges = []
            _count = 0
            real_nodes_id = set()
            for edge_temp_i in range(nodes.shape[0]):
                for edge_temp_j in range(edge_temp_i + 1, nodes.shape[0]):
                    if prob[_count, 1] > 0.8:
                        real_nodes_id.add(edge_temp_i)
                        real_nodes_id.add(edge_temp_j)
                        edges.append((nodes[edge_temp_i], nodes[edge_temp_j]))
                    _count += 1
            real_nodes = nodes[list(real_nodes_id)]
            svg = svg_generate(edges, data_name[:-1], real_nodes, samecolor=True, colors=colors,
                               #image_link=None)
                               image_link= DATAPATH + '/rgb/' + data_name[:-1] + '.jpg')
            svg.saveas('to_send/svg_8/' +data_name[:-1] + '.svg')

if __name__ == '__main__':
    main()
