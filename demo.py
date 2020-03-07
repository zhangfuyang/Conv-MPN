import matplotlib.pyplot as plt
import os
from dataset import Graphdataset
import numpy as np
from config import *
from utils import ensure_folder
from SVG_utils import svg_generate
from model import graphNetwork
from metrics import Metrics
from torch.utils.data import DataLoader
from drn import drn_c_26

no_svg = True
no_render = False
no_metric = False
save_npy = False
restore_npy = False
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
#
#save_folder = "gnn_loop_1"
#checkpoint_name = 'checkpoint_15_2.071'
#
#save_folder = "conv_zero_padding"
#checkpoint_name = 'checkpoint_16_2.025'

if len(threshold) > 1:
    save_metrics_npy = True
    corner_metrics = [[],[]]
    edge_metrics = [[],[]]
    loop_metrics = [[],[]]
else:
    save_metrics_npy = False

def main():
    checkpoint = '{}/{}.tar'.format(save_folder, checkpoint_name)  # model checkpoint
    print('checkpoint: ' + str(checkpoint))
    # Load model
    if not restore_npy:
        checkpoint = torch.load(checkpoint, map_location=device)
        param = checkpoint['model'].state_dict()
        drn = drn_c_26(pretrained=False, image_channels=4)
        drn = torch.nn.Sequential(*list(drn.children())[:-7])
        model = graphNetwork(model_loop_time, drn, gnn=gnn, conv_mpn=conv_mpn,
                             edge_feature_map_channel=edge_feature_channels)
        model.double()
        model.load_state_dict(param, strict=True)
        model = model.to(device)
        print(model)
        model.eval()
    metrics = Metrics()
    metrics.reset()

    DATAPATH='/local-scratch/fuyang/cities_dataset'
    DETCORNERPATH='./corner_detection/result/npy'
    graphdataset = Graphdataset(DATAPATH, DETCORNERPATH, phase='valid', full_connected_init=True,
                              mix_gt=False, demo=True, per_edge=False)
    val_loader = DataLoader(graphdataset, batch_size=1, shuffle=False, num_workers=8)
    ensure_folder(save_folder + '/svg')
    ensure_folder(save_folder + '/' + checkpoint_name)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.33, 1.0]).double().to(device))
    if save_npy:
        ensure_folder(save_folder + '/' + checkpoint_name + '/npy')
    for threshold_ in threshold:
        metrics.reset()
        for idx, data in enumerate(val_loader):
            # Read images
            #if "1554421832.78" not in data['name']:
            #    continue
            print(data['name'])
            if restore_npy:
                preds = np.load(save_folder + '/' + checkpoint_name + '/npy/' + data['name'][0] + '.npy')
            else:
                img = data['img'].to(device)
                edge_index = data['edge_index'][0]
                edge_index = edge_index.to(device)
                edge_masks = data['x'][0].to(device)
                label = data['y'][0].to(device)
                with torch.no_grad():
                    preds = model(img, edge_masks, edge_index)

                loss = criterion(preds, label)
                print(loss.item())
                preds = preds.cpu().numpy()
                if save_npy:
                    np.save(save_folder + '/' + checkpoint_name + '/npy/' + data['name'][0], preds)
            prob = np.exp(preds) / np.sum(np.exp(preds), 1)[..., np.newaxis]
            if no_metric is False:
                annot = graphdataset.get_annot(data['name'][0])

                junc = data['pos'][0][:, [1,0]].numpy()
                lines_on = []
                juncs_on = []
                _count = 0
                for corner_i in range(junc.shape[0]):
                    for corner_j in range(corner_i + 1, junc.shape[0]):
                        if prob[_count, 1] > threshold_:
                        #if preds[_count].argmax() == 1:
                            lines_on.append((corner_i, corner_j))
                            if corner_i not in juncs_on:
                                juncs_on.append(corner_i)
                            if corner_j not in juncs_on:
                                juncs_on.append(corner_j)
                        _count += 1

                metrics.forward(annot, junc, juncs_on, lines_on, data['name'][0])

            if no_render is False:
                img = data['img'][0].cpu().numpy().transpose((1,2,0))
                img_pred = img.copy()
                _count = 0
                plt.imshow(img_pred)
                plt.plot(data['pos'][0][:,1], data['pos'][0][:,0], 'bo')
                for edge_i in range(data['pos'][0].shape[0]):
                    for edge_j in range(edge_i + 1, data['pos'][0].shape[0]):
                        if preds[_count].argmax() == 1:
                            plt.plot([data['pos'][0][edge_i][1], data['pos'][0][edge_j][1]],
                                     [data['pos'][0][edge_i][0], data['pos'][0][edge_j][0]],'r')
                        _count += 1
                plt.show()
                plt.close()

            if no_svg is False:
                # svg
                colors = []
                nodes = data['pos'][0].numpy().astype(np.int)
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
                            colors.append(prob[_count, 1])
                        _count += 1
                real_nodes = nodes[list(real_nodes_id)]
                svg = svg_generate(edges, data['name'][0], real_nodes, samecolor=True, colors=colors,
                                   #image_link=None)
                                   image_link=DATAPATH + '/rgb/' + data['name'][0] + '.jpg')
                svg.saveas(save_folder + '/svg/pred_corner/' + data['name'][0] + '.svg')

        if no_metric is False:
            values = metrics.print_metrics()
            f = open(save_folder + '/' + checkpoint_name + '/metric.txt', 'a')
            f.write('corners - precision: %.3f recall: %.3f f_score: %.3f\n' % (values[0], values[1], values[2]))
            f.write('edges - precision: %.3f recall: %.3f f_score: %.3f\n' % (values[3], values[4], values[5]))
            f.write('loops - precision: %.3f recall: %.3f f_score: %.3f\n' % (values[6], values[7], values[8]))
            f.write('loops_v2 - precision: %.3f recall: %.3f f_score: %.3f\n' % (values[9], values[10], values[11]))
            f.write(checkpoint_name + " threshold:" + str(threshold_) +'\n')
            f.close()
            if save_metrics_npy:
                corner_metrics[0].append(values[0])
                corner_metrics[1].append(values[1])
                edge_metrics[0].append(values[3])
                edge_metrics[1].append(values[4])
                loop_metrics[0].append(values[9])
                loop_metrics[1].append(values[10])
    if save_metrics_npy:
        data = {
            'corner': corner_metrics,
            'edge': edge_metrics,
            'loop': loop_metrics
        }
        np.save(save_folder + '/' + checkpoint_name + '/score', data)


if __name__ == '__main__':
    main()
