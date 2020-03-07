import time
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import random
from dataset import Graphdataset
from model import graphNetwork
from utils import *
import logging
from drn import drn_c_26


def train(epoch, train_loader, model, optimizer, criterion):
    model.train()

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    start = time.time()
    #DATA_NUM = len(train_loader)
    #shuffle_sort = list(range(DATA_NUM))
    #random.shuffle(shuffle_sort)
    model.zero_grad()
    for batch_i, data in enumerate(train_loader):
        # Set device options
        img = data['img'].to(device)
        # Zero gradients
        if not per_edge_classifier:
            edge_index = data['edge_index'][0].to(device)
            if (data['y'][0].shape[0] > 105):
                continue
            label = data['y'][0].to(device)
            edge_masks = data['x'][0].to(device)
            y_hat = model(img, edge_masks, edge_index)
        else:
            edge_masks = data['x'].to(device)
            y_hat = model(img, edge_masks, None)
            label = data['y'].to(device)
        loss = criterion(y_hat, label)
        if not per_edge_classifier:
            loss = loss / interval_training
        loss.backward()

        if (batch_i + 1) % interval_training == 0 or per_edge_classifier:
            optimizer.step()
            model.zero_grad()

        del img
        if not per_edge_classifier:
            del edge_index
        del label
        del edge_masks
        # Keep track of metrics
        if not per_edge_classifier:
            losses.update(loss.item() * interval_training)
        else:
            losses.update(loss.item())

        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if batch_i % print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, batch_i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))


def valid(val_loader, model, criterion):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, data in enumerate(val_loader):
            img = data['img'].to(device)
            if not per_edge_classifier:
                if data['y'][0].shape[0] > 105:
                    continue
                edge_index = data['edge_index'][0].to(device)
                label = data['y'][0].to(device)
                edge_masks = data['x'][0].to(device)
                y_hat = model(img, edge_masks, edge_index)
            else:
                edge_masks = data['x'].to(device)
                y_hat = model(img, edge_masks, None)
                label = data['y'].to(device)
            loss = criterion(y_hat, label)

            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                logging.info('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    return losses.avg


def main():
    DATAPATH='/local-scratch/fza49/cities_dataset'
    DETCORNERPATH='/local-scratch/fza49/nnauata/building_reconstruction/geometry-primitive-detector/det_final'

    train_dataset = Graphdataset(DATAPATH, DETCORNERPATH, phase='train', mix_gt=True, per_edge=per_edge_classifier)
    train_dataset_2 = Graphdataset(DATAPATH, DETCORNERPATH, phase='train', mix_gt=False, per_edge=per_edge_classifier)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    train_dataloader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataset = Graphdataset(DATAPATH, DETCORNERPATH, phase='test', per_edge=per_edge_classifier)
    test_datloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    #backbone
    drn = drn_c_26(pretrained=True, image_channels=4)
    drn = nn.Sequential(*list(drn.children())[:-7])
    model = graphNetwork(model_loop_time, drn, edge_feature_map_channel=edge_feature_channels,
                         gnn=gnn, conv_mpn=conv_mpn)

    model.double()
    model = model.to(device)
    model.change_device()
    if pretrain:
        chechpoint_name = 'checkpoint_25_0.602'
        checkpoint = '{}/{}.tar'.format(save_folder, chechpoint_name)
        checkpoint = 'conv_mpn_loop_1/checkpoint_16_2.025.tar'
        print(checkpoint)
        checkpoint = torch.load(checkpoint, map_location=device)
        param = checkpoint['model'].state_dict()
        model.load_state_dict(param, strict=False)


    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = 100000
    epochs_since_improvement = 0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.33, 1.0]).double().to(device))
    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        if epoch % 3 != 0:
            train(epoch, train_dataloader, model, optimizer, criterion)
        else:
            train(epoch, train_dataloader_2, model, optimizer, criterion)

        # One epoch's validation
        val_loss = valid(test_datloader, model, criterion)
        logging.info('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            logging.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
