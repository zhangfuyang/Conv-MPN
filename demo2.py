import matplotlib.pyplot as plt
from dataset import Graphdataset
import numpy as np
from config import *
from utils import ensure_folder
from SVG_utils import svg_generate
from model import graphNetwork
from metrics import Metrics

no_svg = False
no_render = True
no_metric = False
save_npy = False
restore_npy = True
NAME = "1554860191.91"
#threshold = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
threshold = [0.5]
save_folder = "conv_mpn_loop_3_pretrain_2"
checkpoint_name = 'checkpoint_14_0.916'
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
    #param = checkpoint['model'].state_dict()
    #model = graphNetwork(model_loop_time, no_message_passing=no_gnn,
    #                     no_conv_message_passing=no_cmp, gin=gin,
    #                     edge_feature_map_channel=edge_feature_channels,
    #                     use_unet=use_unet)
    #model.double()
    #model.load_state_dict(param, strict=True)
    #model = model.to(device)
    #print(model)
    #model.eval()
    metrics = Metrics()
    metrics.reset()

    DATAPATH='/local-scratch/fza49/cities_dataset'
    DETCORNERPATH='/local-scratch/fza49/nnauata/building_reconstruction/geometry-primitive-detector/det_final'
    PREDEDGEPATH='/local-scratch/fza49/outdoor/cnnForEdge2/results/50_strenghs_3_final'
    val_loader = Graphdataset(DATAPATH, DETCORNERPATH, PREDEDGEPATH, phase='test', full_connected_init=True,
                              mix_gt=False, demo=True)
    for threshold_ in threshold:
        print('---------------------', threshold_, '--------------------')
        metrics.reset()
        data = val_loader.getbyname(NAME)
        print(data['name'])
        if restore_npy:
            preds = np.load(save_folder + '/' + checkpoint_name + '/npy/' + data['name'] + '.npy')
            feature_map_vis = {}
        if no_svg is False:
            # svg

            colors = []
            nodes = np.array(list(data['annot'].keys())).astype(np.int)
          
            gt = val_loader.get_gt(nodes[:, [1,0]], data['annot'])  
            label = []
            for corner_i in range(nodes.shape[0]):
                for corner_j in range(corner_i + 1, nodes.shape[0]):
                    if (corner_i, corner_j) in gt or (corner_j, corner_i) in gt:
                        label.append(1)
                    else:
                        label.append(0)

            edges = []
            _count = 0
            real_nodes_id = set()
            for edge_temp_i in range(nodes.shape[0]):
                for edge_temp_j in range(edge_temp_i + 1, nodes.shape[0]):
                    if label[_count] == 1:
                        real_nodes_id.add(edge_temp_i)
                        real_nodes_id.add(edge_temp_j)
                        edges.append((nodes[edge_temp_i], nodes[edge_temp_j]))
                    _count += 1
            real_nodes = nodes[list(real_nodes_id)]
            svg = svg_generate(edges, NAME, real_nodes, samecolor=True, colors=colors,
                               #image_link=None)
                               image_link='/local-scratch/fza49/cities_dataset/rgb/' + NAME + '.jpg')
            svg.saveas(NAME + '_gt.svg')




            colors = []
            print(data['pos'])
            nodes = data['pos'].astype(np.int)
            nodes = nodes[:, [1, 0]]
            edges = []
            _count = 0
            real_nodes_id = set()
            for edge_temp_i in range(nodes.shape[0]):
                for edge_temp_j in range(edge_temp_i + 1, nodes.shape[0]):
                    if preds[_count].argmax() == 1:
                        real_nodes_id.add(edge_temp_i)
                        real_nodes_id.add(edge_temp_j)
                        edges.append((nodes[edge_temp_i], nodes[edge_temp_j]))
                    _count += 1
            real_nodes = nodes[list(real_nodes_id)]
            svg = svg_generate(edges, NAME, real_nodes, samecolor=True, colors=colors,
                               #image_link=None)
                               image_link='/local-scratch/fza49/cities_dataset/rgb/' + NAME + '.jpg')
            svg.saveas(NAME + '.svg')

if __name__ == '__main__':
    main()
