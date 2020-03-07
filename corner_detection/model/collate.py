import torch
import numpy as np

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        N = max(map(lambda x: x[4].shape[self.dim], batch))
 
       
        # pad according to max_len
        new_batch = []
        for x, y, z, w, k, n, c, l, m, r in batch:
            nk = np.pad(k, ((0, N-k.shape[0])), 'constant', constant_values=-1)
            nn = np.pad(n, ((0, N-n.shape[0])), 'constant', constant_values=-1)
            nc = np.pad(c, ((0, N-c.shape[0])), 'constant', constant_values=-1)
            nl = np.pad(l, ((0, N-l.shape[0]), (0, 0)), 'constant')
            nm = np.pad(m, ((0, N-m.shape[0]), (0, 0), (0, 0)), 'constant')
            new_batch.append((x, y, z, w, nk, nn, nc, nl, nm, r))

        # stack all
        images = np.stack([x[0] for x in new_batch], axis=0)
        image_metas = np.stack([x[1] for x in new_batch], axis=0)
        rpn_match = np.stack([x[2] for x in new_batch], axis=0)
        rpn_bbox = np.stack([x[3] for x in new_batch], axis=0)
        gt_class_ids = np.stack([x[4] for x in new_batch], axis=0)
        gt_edges_dir = np.stack([x[5] for x in new_batch], axis=0)
        gt_coords = np.stack([x[6] for x in new_batch], axis=0)
        gt_boxes = np.stack([x[7] for x in new_batch], axis=0)
        gt_masks = np.stack([x[8] for x in new_batch], axis=0)
        gt_outline = np.stack([x[9] for x in new_batch], axis=0)

        # convert to torch tensors
        images = torch.from_numpy(images)
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox)
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_edges_dir = torch.from_numpy(gt_edges_dir)
        gt_coords = torch.from_numpy(gt_coords)
        gt_boxes = torch.from_numpy(gt_boxes)
        gt_masks = torch.from_numpy(gt_masks)
        gt_outline = torch.from_numpy(gt_outline)

        return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_edges_dir, gt_coords, gt_boxes, gt_masks, gt_outline

    def __call__(self, batch):
        return self.pad_collate(batch)