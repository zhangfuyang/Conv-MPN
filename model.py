import torch.nn as nn
from config import *
from unet import UNet
from torch.nn.parameter import Parameter
import math


class graphNetwork(nn.Module):
    def __init__(self, times, backbone, edge_feature_map_channel=32,
                 conv_mpn=False, gnn=False):
        super(graphNetwork, self).__init__()
        self.edge_feature_channel = edge_feature_map_channel
        self.rgb_net = nn.Sequential(
            backbone,
            nn.Conv2d(2 * self.edge_feature_channel, self.edge_feature_channel, kernel_size=3, stride=1, padding=1)
        )
        self.gnn = gnn
        self.times = times
        self.conv_mpn = conv_mpn
        # gnn baseline
        self.vector_size = 16 * self.edge_feature_channel
        if gnn:
            vector_size = self.vector_size
            self.loop_net = nn.ModuleList([nn.Sequential(
                nn.Conv2d(2 * vector_size, 2 * vector_size, kernel_size=1, stride=1),
                nn.BatchNorm2d(2 * vector_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * vector_size, 2 * vector_size, kernel_size=1, stride=1),
                nn.BatchNorm2d(2 * vector_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * vector_size, 2 * vector_size, kernel_size=1, stride=1),
                nn.BatchNorm2d(2 * vector_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * vector_size, vector_size, kernel_size=1, stride=1),
                nn.BatchNorm2d(vector_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(vector_size, vector_size, kernel_size=1, stride=1),
                nn.BatchNorm2d(vector_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(vector_size, vector_size, kernel_size=1, stride=1),
                nn.BatchNorm2d(vector_size),
                nn.ReLU(inplace=True)
            ) for _ in range(self.times)])

        if conv_mpn:
            self.loop_net = nn.ModuleList([
                conv_mpn_model(2 * self.edge_feature_channel,
                               self.edge_feature_channel)
                for _ in range(self.times)])

        self.edge_pred_layer = nn.Sequential(
            nn.Conv2d(self.edge_feature_channel, self.edge_feature_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.edge_feature_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.edge_feature_channel, 2 * self.edge_feature_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * self.edge_feature_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.edge_feature_channel, 2 * self.edge_feature_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * self.edge_feature_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.edge_feature_channel, 4 * self.edge_feature_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.edge_feature_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * self.edge_feature_channel, 4 * self.edge_feature_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.edge_feature_channel),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.AdaptiveAvgPool2d((2,2))
        self.fc = nn.Linear(self.vector_size, 2)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.track_running_stats=False

    def change_device(self):
        self.rgb_net.to(device)
        self.loop_net.to(device2)
        self.edge_pred_layer.to(device2)
        self.fc.to(device)

    def forward(self, img, edge_masks, edge_index=None):
        if self.training is False:
            tt = math.ceil(edge_masks.shape[0] / 105)
            edge_feature_init = torch.zeros((edge_masks.shape[0], self.edge_feature_channel, 64, 64)).double().to(device)
            for time in range(tt):
                if time == tt - 1:
                    edge_sub_masks = edge_masks[time * 105:, :, :]
                else:
                    edge_sub_masks = edge_masks[time * 105:(time+1) * 105, :, :]
                img_expand = img.expand(edge_sub_masks.shape[0], -1, -1, -1)
                feature_in = torch.cat((img_expand, edge_sub_masks.unsqueeze(1)), 1)
                if time == tt - 1:
                    edge_feature_init[time * 105:] = self.rgb_net(feature_in)
                else:
                    edge_feature_init[time*105:(time+1)*105] = self.rgb_net(feature_in)
                del feature_in
        else:
            img = img.expand(edge_masks.shape[0], -1, -1, -1)
            feature_in = torch.cat((img, edge_masks.unsqueeze(1)), 1)
            edge_feature_init = self.rgb_net(feature_in)
        edge_feature = edge_feature_init
        if device != device2:
            edge_feature = edge_feature.to(device2)
        if self.conv_mpn:
            for t in range(self.times):
                feature_neighbor = torch.zeros_like(edge_feature)
                for edge_iter in range(edge_masks.shape[0]):
                    feature_temp = edge_feature[edge_index[1, torch.where(edge_index[0,:] == edge_iter)[0]]]
                    feature_neighbor[edge_iter] = torch.max(feature_temp, 0)[0]
                edge_feature = torch.cat((edge_feature, feature_neighbor), 1)
                edge_feature = self.loop_net[t](edge_feature)
        if self.training is False:
            tt = math.ceil(edge_masks.shape[0] / 105)
            edge_pred = torch.zeros((edge_masks.shape[0], 4*self.edge_feature_channel, 64, 64)).double().to(device)
            for time in range(tt):
                if time == tt - 1:
                    edge_sub_feature = edge_feature[time * 105:, :, :]
                else:
                    edge_sub_feature = edge_feature[time * 105:(time+1) * 105, :, :]
                if time == tt - 1:
                    edge_pred[time * 105:] = self.edge_pred_layer(edge_sub_feature)
                else:
                    edge_pred[time*105:(time+1)*105] = self.edge_pred_layer(edge_sub_feature)
                del edge_sub_feature
        else:
            edge_pred = self.edge_pred_layer(edge_feature)
        edge_pred = self.maxpool(edge_pred)
        edge_pred = edge_pred.view((edge_masks.shape[0], self.vector_size, 1, 1))
        if self.gnn:
            for t in range(self.times):
                feature_neighbor = torch.zeros_like(edge_pred)
                for edge_iter in range(edge_masks.shape[0]):
                    feature_temp = edge_pred[edge_index[1, torch.where(edge_index[0,:] == edge_iter)[0]]]
                    feature_neighbor[edge_iter] = torch.max(feature_temp, 0)[0]
                edge_pred = torch.cat((edge_pred, feature_neighbor), 1)
                edge_pred = self.loop_net[t](edge_pred)
        edge_pred = torch.flatten(edge_pred, 1)
        if device != device2:
            edge_pred = edge_pred.to(device)
        fc = self.fc(edge_pred)
        return fc


class conv_mpn_model(nn.Module):
    def __init__(self, inchannels, out_channels):
        super(conv_mpn_model, self).__init__()
        assert inchannels >= out_channels
        self.out_channels = out_channels
        self.seq = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inchannels, track_running_stats=True),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inchannels, track_running_stats=True),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inchannels, track_running_stats=True),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inchannels, track_running_stats=True),
            nn.Conv2d(inchannels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, track_running_stats=True)
        )

    def forward(self, x):
        return self.seq(x)

