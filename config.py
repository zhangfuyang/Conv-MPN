import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

input_nbr = 3
lr = 0.0005
patience = 50
start_epoch = 0
epochs = 120
print_freq = 20
interval_training = 8
save_folder = 'conv_mpn_loop_3_pretrain_2'
model_loop_time = 3
edge_feature_channels = 32
conv_mpn = True
gnn = False
pretrain = False
per_edge_classifier = not gnn and not conv_mpn
batch_size = 1 if not per_edge_classifier else 32
new_label = True
