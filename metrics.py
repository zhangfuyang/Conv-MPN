import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from collections import defaultdict
from intersections import doIntersect


class Metrics(): 
    def __init__(self):

        # corners
        self.curr_corner_tp = 0.0
        self.curr_corner_fp = 0.0
        self.n_corner_samples = 0.0
        self.per_corner_sample_score = {}

        # edges
        self.curr_edge_tp = 0.0
        self.curr_edge_fp = 0.0
        self.n_edge_samples = 0.0
        self.per_edge_sample_score = {}

        # loops
        self.curr_loop_tp = 0.0
        self.curr_loop_fp = 0.0
        self.n_loop_samples = 0.0
        self.per_loop_sample_score = {}

        # loops V2
        self.curr_loop_tp_v2 = 0.0
        self.curr_loop_fp_v2 = 0.0
        self.n_loop_samples_v2 = 0.0
        self.per_loop_sample_score_v2 = {}

        # loops V3
        self.curr_loop_tp_v3 = 0.0
        self.curr_loop_fp_v3 = 0.0
        self.n_loop_samples_v3 = 0.0
        self.per_loop_sample_score_v3 = {}

    def calc_corner_metrics(self):
        recall = self.curr_corner_tp/(self.n_corner_samples+1e-8)
        precision = self.curr_corner_tp/(self.curr_corner_tp+self.curr_corner_fp+1e-8)
        return recall, precision

    def calc_edge_metrics(self):
        recall = self.curr_edge_tp/(self.n_edge_samples+1e-8)
        precision = self.curr_edge_tp/(self.curr_edge_tp+self.curr_edge_fp+1e-8)
        return recall, precision

    def calc_loop_metrics(self):
        recall = self.curr_loop_tp/(self.n_loop_samples+1e-8)
        precision = self.curr_loop_tp/(self.curr_loop_tp+self.curr_loop_fp+1e-8)
        return recall, precision

    def calc_loop_metrics_v2(self):
        recall = self.curr_loop_tp_v2/(self.n_loop_samples_v2+1e-8)
        precision = self.curr_loop_tp_v2/(self.curr_loop_tp_v2+self.curr_loop_fp_v2+1e-8)
        return recall, precision

    def calc_loop_metrics_v3(self):
        recall = self.curr_loop_tp_v3/(self.n_loop_samples_v3+1e-8)
        precision = self.curr_loop_tp_v3/(self.curr_loop_tp_v3+self.curr_loop_fp_v3+1e-8)
        return recall, precision

    def print_metrics(self):

        # print scores
        values = []
        recall, precision = self.calc_corner_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]
        print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        # print scores
        recall, precision = self.calc_edge_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]        
        print('edges - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        recall, precision = self.calc_loop_metrics()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]
        print('loops - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        recall, precision = self.calc_loop_metrics_v2()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]
        print('loops_v2 - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))

        recall, precision = self.calc_loop_metrics_v3()
        f_score = 2.0*precision*recall/(precision+recall+1e-8)
        values += [precision, recall, f_score]
        print('loops_v3 - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
        return values

    def reset(self):

        # corners
        self.curr_corner_tp = 0.0
        self.curr_corner_fp = 0.0
        self.n_corner_samples = 0.0
        self.per_corner_sample_score = {}

        # edges
        self.curr_edge_tp = 0.0
        self.curr_edge_fp = 0.0
        self.n_edge_samples = 0.0
        self.per_edge_sample_score = {}

        # loops
        self.curr_loop_tp = 0.0
        self.curr_loop_fp = 0.0
        self.n_loop_samples = 0.0
        self.per_loop_sample_score = {}

        # loops_v2
        self.curr_loop_tp_v2 = 0.0
        self.curr_loop_fp_v2 = 0.0
        self.n_loop_samples_v2 = 0.0
        self.per_loop_sample_score_v2 = {}

        # loops V3
        self.curr_loop_tp_v3 = 0.0
        self.curr_loop_fp_v3 = 0.0
        self.n_loop_samples_v3 = 0.0
        self.per_loop_sample_score_v3 = {}
        return

    def forward(self, graph_gt, junctions, juncs_on, lines_on, _id, thresh=8.0, iou_thresh=0.7):

        ## Compute corners precision/recall
        gts = np.array([list(x) for x in graph_gt])

        if len(juncs_on) > 0:
            dets = np.array(junctions)[juncs_on]
        else:
            dets = np.array([])

        per_sample_corner_tp = 0.0
        per_sample_corner_fp = 0.0
        found = [False] * gts.shape[0]
        c_det_annot = {}

        # for each corner detection
        for i, det in enumerate(dets):

            # get closest gt
            near_gt = [0, 9999999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt-det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt] 

            # hit (<= thresh) and not found yet 
            if near_gt[1] <= thresh and not found[near_gt[0]]:
                per_sample_corner_tp += 1.0
                found[near_gt[0]] = True
                c_det_annot[juncs_on[i]] = near_gt[0]

            # not hit or already found
            else:
                per_sample_corner_fp += 1.0

        # update counters for corners
        self.curr_corner_tp += per_sample_corner_tp
        self.curr_corner_fp += per_sample_corner_fp
        self.n_corner_samples += gts.shape[0]
        self.per_corner_sample_score.update({_id: {'recall': per_sample_corner_tp/gts.shape[0], 'precision': per_sample_corner_tp/(per_sample_corner_tp+per_sample_corner_fp+1e-8)}}) 

        ## Compute edges precision/recall
        per_sample_edge_tp = 0.0
        per_sample_edge_fp = 0.0
        edge_corner_annots = edges_from_annots(graph_gt)

        # for each detected edge 
#         print(lines_on)
        for l, e_det in enumerate(lines_on):
            c1, c2 = e_det
            
            # check if corners are mapped
            if (c1 not in c_det_annot.keys()) or (c2 not in c_det_annot.keys()):
                per_sample_edge_fp += 1.0                
                continue

            # check hit
            c1_prime = c_det_annot[c1]
            c2_prime = c_det_annot[c2]
            is_hit = False
            
            for k, e_annot in enumerate(edge_corner_annots):
                c3, c4 = e_annot
                if ((c1_prime == c3) and (c2_prime == c4)) or ((c1_prime == c4) and (c2_prime == c3)):
                    is_hit = True

            # hit
            if is_hit == True:
                per_sample_edge_tp += 1.0
            # not hit 
            else:
                per_sample_edge_fp += 1.0

        # update counters for edges
        self.curr_edge_tp += per_sample_edge_tp
        self.curr_edge_fp += per_sample_edge_fp
        self.n_edge_samples += edge_corner_annots.shape[0]
        self.per_edge_sample_score.update({_id: {'recall': per_sample_edge_tp/edge_corner_annots.shape[0], \
            'precision': per_sample_edge_tp/(per_sample_edge_tp+per_sample_edge_fp+1e-8)}}) 

        ## Compute loops precision/recall
        per_sample_loop_tp = 0.0
        per_sample_loop_fp = 0.0

        corners_annots = np.array([list(x) for x in graph_gt])
        pred_edge_map = draw_edges(lines_on, junctions)
        pred_edge_map = fill_regions(pred_edge_map)
        annot_edge_map = draw_edges(edge_corner_annots, corners_annots)
        annot_edge_map = fill_regions(annot_edge_map)

        pred_rs = extract_regions(pred_edge_map)
        annot_rs = extract_regions(annot_edge_map)

        # for each predicted region
        found = [False] * len(annot_rs)
        for i, r_det in enumerate(pred_rs):

            # get closest gt
            near_gt = [0, 0, None]
            for k, r_gt in enumerate(annot_rs):
                iou = np.logical_and(r_gt, r_det).sum()/float(np.logical_or(r_gt, r_det).sum())
                # print(i, k, iou)
                if iou > near_gt[1]:
                    near_gt = [k, iou, r_gt] 

            # hit (<= thresh) and not found yet 
            if near_gt[1] >= iou_thresh and not found[near_gt[0]]:
                per_sample_loop_tp += 1.0
                found[near_gt[0]] = True
                
            # not hit or already found
            else:
                per_sample_loop_fp += 1.0
        
        # update counters for corners
        self.curr_loop_tp += per_sample_loop_tp
        self.curr_loop_fp += per_sample_loop_fp
        self.n_loop_samples += len(annot_rs)
        self.per_loop_sample_score.update({_id: {'recall': per_sample_loop_tp/len(annot_rs), 'precision': per_sample_loop_tp/(per_sample_loop_tp+per_sample_loop_fp+1e-8)}})


        ###########################################################################################################
        # Compute loop metrics V2
        per_sample_loop_tp_v2 = 0.0
        per_sample_loop_fp_v2 = 0.0

        corners_annots = np.array([list(x) for x in graph_gt])
        annot_rs_v2, _ = extract_regions_v2(corners_annots, np.ones(corners_annots.shape[0]), edge_corner_annots)
        pred_rs_v2, pred_fp = extract_regions_v2(junctions, juncs_on, lines_on)
        found = [False] * len(annot_rs_v2)
        for i, r_det in enumerate(pred_rs_v2):

            # if loop contains intersection there is no need to check it's FP
            if i in pred_fp:
                per_sample_loop_fp_v2 += 1.0
                continue

            # get closest gt
            near_gt = [0, 0, None]
            for k, r_gt in enumerate(annot_rs_v2):
                iou = np.logical_and(r_gt, r_det).sum()/float(np.logical_or(r_gt, r_det).sum())
                # print(i, k, iou)
                if iou > near_gt[1]:
                    near_gt = [k, iou, r_gt] 

            # # debug
            # im_arr1 = np.zeros((256, 256, 3))
            # im_arr2 = np.zeros((256, 256, 3))
            # # print(np.logical_and(r_det, r_det).sum()/float(np.logical_or(r_det, r_det).sum()))

            # inds1 = np.array(np.where(r_det>0))
            # im_arr1[inds1[0, :], inds1[1, :]] = [0, 0, 255]

            # inds2 = np.array(np.where(near_gt[2]>0))
            # im_arr2[inds2[0, :], inds2[1, :]] = [255, 0, 0]

            # print(near_gt[1])
            # im_deb1 = Image.fromarray(im_arr1.astype('uint8'))
            # im_deb2 = Image.fromarray(im_arr2.astype('uint8'))

            # plt.figure()
            # plt.imshow(im_deb1)
            # plt.figure()
            # plt.imshow(im_deb2)
            # plt.show()

            # hit (<= thresh) and not found yet 
            if near_gt[1] >= iou_thresh and not found[near_gt[0]]:
                per_sample_loop_tp_v2 += 1.0
                found[near_gt[0]] = True
                
            # not hit or already found
            else:
                per_sample_loop_fp_v2 += 1.0

        # update counters for corners
        self.curr_loop_tp_v2 += per_sample_loop_tp_v2
        self.curr_loop_fp_v2 += per_sample_loop_fp_v2
        self.n_loop_samples_v2 += len(annot_rs_v2)
        print('tp:{} fp: {}, tot:{}'.format(per_sample_loop_tp, per_sample_loop_fp, len(annot_rs)))
        print('tp_v2:{} fp_v2: {}, tot_v2:{}'.format(per_sample_loop_tp_v2, per_sample_loop_fp_v2, len(annot_rs_v2)))
        self.per_loop_sample_score_v2.update({_id: {'recall': per_sample_loop_tp_v2/len(annot_rs_v2), 'precision': per_sample_loop_tp_v2/(per_sample_loop_tp_v2+per_sample_loop_fp_v2+1e-8)}})


        ###########################################################################################################
        # Compute loop metrics V3
        per_sample_loop_tp_v3 = 0.0
        per_sample_loop_fp_v3 = 0.0

        corners_annots = np.array([list(x) for x in graph_gt])
        annot_rs_v3, _ = extract_regions_v3(corners_annots, np.ones(corners_annots.shape[0]), edge_corner_annots)
        pred_rs_v3, pred_fp = extract_regions_v3(junctions, juncs_on, lines_on)
        found = [False] * len(annot_rs_v3)
        for i, r_det in enumerate(pred_rs_v3):
            # if loop contains intersection there is no need to check it's FP
            if i in pred_fp:
                per_sample_loop_fp_v3 += 1.0
                continue

            # get mathcing gt
            A = sorted(set(r_det))
            is_real = False
            for gt in annot_rs_v3:
                if len(gt) != len(r_det):
                    continue
                B = sorted(set(gt))
                region_match = True
                for node_i in range(len(B)):
                    if (A[node_i][0] - B[node_i][0])**2+(A[node_i][1] - B[node_i][1])**2 >= thresh*thresh:
                        region_match = False
                        break
                if region_match:
                    is_real = True
            if is_real:
                per_sample_loop_tp_v3 += 1.0
            else:
                per_sample_loop_fp_v3 += 1.0

        self.curr_loop_tp_v3 += per_sample_loop_tp_v3
        self.curr_loop_fp_v3 += per_sample_loop_fp_v3
        self.n_loop_samples_v3 += len(annot_rs_v3)
        self.per_loop_sample_score_v3.update({_id: {'recall': per_sample_loop_tp_v3/len(annot_rs_v3), 'precision': per_sample_loop_tp_v3/(per_sample_loop_tp_v3+per_sample_loop_fp_v3+1e-8)}})
        print('tp_v3:{} fp_v3: {}, tot_v3:{}'.format(per_sample_loop_tp_v3, per_sample_loop_fp_v3, len(annot_rs_v3)))




        return

def draw_edges(edge_corner, corners, mode="det"):

    im = Image.new('L', (256, 256))
    draw = ImageDraw.Draw(im)
    for e in edge_corner:
        c1, c2 = e
        if "annot" in mode:
            y1, x1, _, _ = corners[c1]
            y2, x2, _, _ = corners[c2]
        elif "det" in mode:
            y1, x1 = corners[c1]
            y2, x2 = corners[c2]
        draw.line((x1, y1, x2, y2), width=3, fill='white')

    # import matplotlib.pyplot as plt
    # plt.imshow(im)
    # plt.show(im)
    return np.array(im)

def edges_from_annots(graph):

    # map corners
    corner_set = []
    corner_xy_map = {}
    for k, v in enumerate(graph):
        x, y = v
        corner = [x, y]
        corner_set.append(corner)
        corner_xy_map[(x, y)] = k

    # map edges
    edge_set = []
    edge_map = {}
    count = 0
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

            edge = (corner_xy_map[(x1, y1)], corner_xy_map[(x2, y2)])
            if edge not in edge_map:
                edge_map[edge] = count
                edge_set.append(edge)
                count += 1

    edge_set = np.array([list(e) for e in edge_set])
    return np.array(edge_set)

def extract_regions(region_mask):
    inds = np.where((region_mask > 1) & (region_mask < 255))
    tags = set(region_mask[inds])
    tag_depth = dict()
    rs = []
    for t in tags:
        if t > 0:
            r = np.zeros_like(region_mask)
            inds = np.where(region_mask == t)
            r[inds[1], inds[0]] = 1
            if r[0][0] == 0 and r[0][-1] == 0 and r[-1][0] == 0 and r[-1][-1] == 0:
                rs.append(r)
                pass
    
    # # debug
    # for r in rs:
    #     im = Image.fromarray(r*255.0)
    #     plt.imshow(im)
    #     plt.show()
    return rs


# def getAngle(pt1, pt2, pt3):

#     # return angle in clockwise direction
#     x, y = pt1
#     xn, yn = pt3
#     dx, dy = xn-x, yn-y
#     dir_x, dir_y = (dx, dy)/(np.linalg.norm([dx, dy])+1e-8)
#     rad = np.arctan2(-dir_y, dir_x)
#     ang1 = np.degrees(rad)

#     x, y = pt2
#     xn, yn = pt3
#     dx, dy = xn-x, yn-y
#     dir_x, dir_y = (dx, dy)/(np.linalg.norm([dx, dy])+1e-8)
#     rad = np.arctan2(-dir_y, dir_x)
#     ang2 = np.degrees(rad)

#     return  ang1-ang2
#     # if ang < 0:
#     #     ang = (ang + 360) % 360
#     # return 360-ang

def getAngle(pt1, pt2, pt3):

    # return angle in clockwise direction
    x1, y1 = (pt1-pt2)
    x2, y2 = (pt3-pt2)
    x1, y1 = (x1, y1)/np.linalg.norm([x1, y1])
    x2, y2 = (x2, y2)/np.linalg.norm([x2, y2])
    rad = np.arctan2(y2, x2)-np.arctan2(y1, x1)
    ang = np.degrees(rad)
    if ang < 0:
        ang += 360
    return  ang

def get_sequence(edges, junctions, start_corner):
    poly = [start_corner]
    for i, j in edges[1:]:
        if i in poly:
            poly.append(j)
        else:
            poly.append(i)
    return poly + [poly[0]]

def is_clockwise(edges, junctions, start_corner):
    poly = [junctions[start_corner]]
    keep_track = [start_corner]
    for i, j in edges[1:]:
        if i in keep_track:
            poly.append(junctions[j])
            keep_track.append(j)
        else:
            poly.append(junctions[i])
            keep_track.append(i)
    
    poly_shifted = np.array([poly[-1]] + poly[:-1])
    poly = np.array(poly)

    ### DEBUG
    # print(keep_track)
    # print(np.sum((poly_shifted[:, 0]-poly[:, 0])*(poly_shifted[:, 1]+poly[:, 1])))
    # im_deb = Image.new('RGB', (256, 256))
    # dr = ImageDraw.Draw(im_deb)
    # for k, (x, y) in enumerate(poly):
    #   dr.text((x, y), str(k),(0, 255, 0))
    # plt.figure()
    # plt.imshow(im_deb)
    # plt.show()

    if np.sum((poly_shifted[:, 0]-poly[:, 0])*(poly_shifted[:, 1]+poly[:, 1])) > 0:
        return True
    return False

def sort_clockwise(junctions, curr_corner, curr_edge, edges):
    
    if len(edges) == 0:
        return None

    # compute outcoming angles
    out_angles = []
    prev_corner = curr_edge[1] if curr_corner == curr_edge[0] else curr_edge[0]
    for i, j in edges:
        next_corner = i if curr_corner == j else j
        out_angles.append(getAngle(junctions[prev_corner], junctions[curr_corner], junctions[next_corner])) 
    next_edge_ind = sorted(range(len(out_angles)), key=lambda k: out_angles[k], reverse=True)[0]

    # ### DEBUG
    # im_deb = Image.new('RGB', (256, 256))
    # dr = ImageDraw.Draw(im_deb)
    # x0, y0 = junctions[curr_edge[0]]
    # x1, y1 = junctions[curr_edge[1]]
    # dr.line((x0, y0, x1, y1), width=3, fill='blue')
    # for i, j in edges:
    #     x0, y0 = junctions[i]
    #     x1, y1 = junctions[j]
    #     dr.line((x0, y0, x1, y1), width=3, fill='red')
    # next_edge = edges[next_edge_ind]
    # x0, y0 = junctions[next_edge[0]]
    # x1, y1 = junctions[next_edge[1]]
    # dr.line((x0, y0, x1, y1), width=3, fill='green')
    # for k , (x, y) in enumerate(junctions):
    #     dr.text((x, y), str(k),(0, 255, 0))
    # plt.imshow(im_deb)
    # plt.show()

    return next_edge_ind

def check_polygon(edges, degree_list=False):
    degs = defaultdict(int)
    for e in edges:
        c1 = e[0]
        c2 = e[1]
        degs[c1] += 1
        degs[c2] += 1
    for k in degs.keys():
        if degs[k] < 2:
            if degree_list:
                return False, degs
            return False
    if degree_list:
        return True, degs
    return True

def find_loop(junctions, curr_edge, start_corner, lines_on, tracker, is_contour=False):

    visited_edges = [curr_edge]
    curr_corner = start_corner
    is_broken = False
    while check_polygon(visited_edges) == False:
        # print(visited_edges, check_polygon(visited_edges))
        # print(curr_edge, curr_corner)

        # sort not visited edges and pick closest one
        not_visited = [x for x in lines_on if x not in visited_edges]

        # pick next edge candidates
        candidates = [[], []]
        # print(not_visited)
        for (p1, p2) in not_visited:
            if (p1 == curr_corner):
                candidates[0].append((p1, p2))
                candidates[1].append([p1, p2])
            elif (p2 == curr_corner):
                candidates[0].append((p2, p1))
                candidates[1].append((p1, p2))

        # sort edges clockwise
        next_edge_ind = sort_clockwise(junctions, curr_corner, curr_edge, candidates[0])
        if next_edge_ind is None:
            is_broken = True
            break
        else:
            # if no visited edge stop and return loop
            next_edge = candidates[1][next_edge_ind]
            visited_edges.append(tuple(next_edge))
            curr_corner = next_edge[0] if curr_corner == next_edge[1] else next_edge[1]
            curr_edge = next_edge
    poly = get_sequence(visited_edges, junctions, start_corner)

    # ############## DEBUG
    # print(poly)
    # print(tracker)
    # im_deb = Image.new('RGB', (256, 256))
    # dr = ImageDraw.Draw(im_deb)
    # for i, j in visited_edges:
    #     x0, y0 = junctions[i]
    #     x1, y1 = junctions[j]
    #     dr.line((x0, y0, x1, y1), width=3, fill='blue')
    # for k , (x, y) in enumerate(junctions):
    #     dr.text((x, y), str(k),(0, 255, 0))
    # # check if is repeated
    # is_repeated = True
    # for k in range(len(poly)-1):
    #     if tuple([poly[k], poly[k+1]]) not in tracker:
    #         is_repeated = False
    # print('is_contour:{} is_repeated:{}, is_clockwise:{}, is_broken:{}'.format(is_contour, is_repeated, is_clockwise(visited_edges, junctions, start_corner), is_broken))
    # plt.imshow(im_deb)
    # plt.show()
        
    if is_clockwise(visited_edges, junctions, start_corner) and not is_broken:

        # check if is repeated
        is_repeated = True
        for k in range(len(poly)-1):
            if tuple([poly[k], poly[k+1]]) not in tracker:
                is_repeated = False

        # update tracker with visited edges
        if not is_repeated:
            # init new tracker
            updated_tracker = tracker.copy()
            for k in range(len(poly)-1):
                updated_tracker.add(tuple([poly[k], poly[k+1]]))
            return poly, updated_tracker, False
        else:
            return poly, tracker.copy(), True
    else:
        if is_contour:
            updated_tracker = tracker.copy()
            for k in range(len(poly)-1):
                updated_tracker.add(tuple([poly[k], poly[k+1]]))
            return poly, updated_tracker, False
        return None, tracker.copy(), True
    

def find_contour(junctions, lines_on, tracker):

    # get right most corner
    max_x = -1
    min_y = -1
    max_j = -1
    for k, (x, y) in enumerate(junctions):
        if max_x < x:
            min_y = y
            max_x = x
            max_j = k
        elif max_x == x:
            if min_y > y:
                min_y = y
                max_x = x
                max_j = k 
    curr_corner = max_j

    # get edge in contour
    candidates = [[], []] 
    for (p1, p2) in lines_on:
        if (p1 == curr_corner):
            candidates[0].append((p1, p2))
            candidates[1].append([p1, p2])
        elif (p2 == curr_corner):
            candidates[0].append((p2, p1))
            candidates[1].append((p1, p2))            

    # sort angles
    junc_prev = np.array(junctions[curr_corner])
    junc_prev[1] -= 50.0
    out_angles = []
    for i, j in candidates[0]:
        next_corner = i if curr_corner == j else j
        out_angles.append(getAngle(junc_prev, junctions[curr_corner], junctions[next_corner])) 
        next_edge_ind = sorted(range(len(out_angles)), key=lambda k: out_angles[k], reverse=True)[0]
    next_edge = candidates[1][next_edge_ind]
    start_corner = next_edge[1] if curr_corner == next_edge[0] else next_edge[0]

    # ### DEBUG
    # print(out_angles)
    # im_deb = Image.new('RGB', (256, 256))
    # dr = ImageDraw.Draw(im_deb)
    # x0, y0 = junc_prev
    # x1, y1 = junctions[curr_corner]
    # dr.line((x0, y0, x1, y1), width=3, fill='blue')
    # for i, j in candidates[0]:
    #     x0, y0 = junctions[i]
    #     x1, y1 = junctions[j]
    #     dr.line((x0, y0, x1, y1), width=3, fill='red')
    # next_edge = candidates[0][next_edge_ind]
    # x0, y0 = junctions[next_edge[0]]
    # x1, y1 = junctions[next_edge[1]]
    # dr.line((x0, y0, x1, y1), width=3, fill='green')
    # for k , (x, y) in enumerate(junctions):
    #     dr.text((x, y), str(k),(0, 255, 0))
    # plt.figure()
    # plt.imshow(im_deb)
    # plt.show()

    poly, updated_tracker, _ = find_loop(junctions, next_edge, start_corner, lines_on, tracker, is_contour=True)

    # ### DEBUG - contour
    # im_deb = Image.new('RGB', (256, 256))
    # dr = ImageDraw.Draw(im_deb)
    # for k in range(len(poly)-1):
    #     x0, y0 = junctions[poly[k]]
    #     x1, y1 = junctions[poly[k+1]]
    #     dr.line((x0, y0, x1, y1), width=3, fill='magenta')
    # for k , (x, y) in enumerate(junctions):
    #     dr.text((x, y), str(k),(0, 255, 0))
    # print(updated_tracker)
    # plt.imshow(im_deb)
    # plt.show()

    return updated_tracker

def remove_dangling(junctions, juncs_on, lines_on):

    # remove dangling edges
    filtered_edges = list(lines_on)
    is_closed, degree_list = check_polygon(filtered_edges, degree_list=True)
    while (is_closed == False) and (len(filtered_edges) > 0):
        for i in degree_list:
            if degree_list[i] == 1:
                to_remove = None
                for n, (k, l) in enumerate(filtered_edges):
                    if (i == l) or (i == k):
                        to_remove = n
                        break
                if to_remove is not None:
                    filtered_edges.pop(n)
        is_closed, degree_list = check_polygon(filtered_edges, degree_list=True)

    # remove junctions with degree zero
    new_junctions = []
    tracker = []
    old_to_new_map = dict()
    for k, l in filtered_edges:
        if k not in tracker:
            tracker.append(k)
            old_to_new_map[k] = len(new_junctions)
            new_junctions.append(junctions[k])
        if l not in tracker:
            tracker.append(l)
            old_to_new_map[l] = len(new_junctions)
            new_junctions.append(junctions[l])    

    # remap edges
    new_lines_on = np.array([(old_to_new_map[i], old_to_new_map[j]) for i, j in filtered_edges])


    return np.array(new_junctions), np.ones(junctions.shape[0]), new_lines_on

def extract_regions_v2(junctions, juncs_on, lines_on):
    
    # is_closed, degree_list = check_polygon(lines_on, degree_list=True)
    # print(is_closed, degree_list)

    # handle intersections
    edges_intersect = set()
    for k, (p1, q1) in enumerate(lines_on):
        for l, (p2, q2) in enumerate(lines_on):
            if k > l:
                if doIntersect(junctions[p1], junctions[q1], junctions[p2], junctions[q2]):
                    edges_intersect.add((p1, q1))
                    edges_intersect.add((p2, q2))

    # remove dangling edges
    junctions, juncs_on, lines_on = remove_dangling(junctions, juncs_on, lines_on)

    # init variables
    lines_on = [tuple([x, y]) for x, y in lines_on]
    tracker = set() # (clockwise, anticlockwise)

    if len(junctions) == 0:
        return [], []

    # find contour
    tracker = find_contour(junctions, lines_on, tracker)

    # extract regions
    start_edge = lines_on[0]
    regions = []
    while True: 

        # find loop, updating tracker
        region, updated_tracker, is_repeated = find_loop(junctions, start_edge, start_edge[1], lines_on, tracker)
        if region is None:
            region, updated_tracker, is_repeated = find_loop(junctions, start_edge, start_edge[0], lines_on, tracker)
        tracker = updated_tracker
        if not is_repeated:
            regions.append(region)

        # find next edge that does not appear twice in tracker
        start_edge = None
        for (i, j) in lines_on:
            if ((i, j) not in tracker):
                start_edge = (i, j)
                tracker.add((i, j))
                break
            if ((j, i) not in tracker):
                start_edge = (j, i)
                tracker.add((j, i))
                break

        # else stop
        if start_edge is None:
            break

    # mark regions using edges with intersection
    false_pos = []
    for l, poly in enumerate(regions):
        for k in range(len(poly)-1):
            i, j = poly[k], poly[k+1]
            if ((i, j) in edges_intersect) or ((j, i) in edges_intersect):
               false_pos.append(l)
               break

    # extract region masks
    region_mks = []
    for poly in regions:
        rm = Image.new('L', (256, 256))
        dr = ImageDraw.Draw(rm)
        poly_coords = [tuple(junctions[x]) for x in poly]
        dr.polygon(poly_coords, fill='white')
        region_mks.append(np.array(rm)/255.0)

    #     ### DEBUG
    #     print(false_pos)
    #     print(poly_coords)
    #     plt.imshow(rm)
    #     plt.show()
    # print('test:', len(region_mks))

    return region_mks, false_pos

def extract_regions_v3(junctions, juncs_on, lines_on):

    # is_closed, degree_list = check_polygon(lines_on, degree_list=True)
    # print(is_closed, degree_list)

    # handle intersections
    edges_intersect = set()
    for k, (p1, q1) in enumerate(lines_on):
        for l, (p2, q2) in enumerate(lines_on):
            if k > l:
                if doIntersect(junctions[p1], junctions[q1], junctions[p2], junctions[q2]):
                    edges_intersect.add((p1, q1))
                    edges_intersect.add((p2, q2))

    # remove dangling edges
    junctions, juncs_on, lines_on = remove_dangling(junctions, juncs_on, lines_on)

    # init variables
    lines_on = [tuple([x, y]) for x, y in lines_on]
    tracker = set() # (clockwise, anticlockwise)

    if len(junctions) == 0:
        return [], []

    # find contour
    tracker = find_contour(junctions, lines_on, tracker)

    # extract regions
    start_edge = lines_on[0]
    regions = []
    while True:

        # find loop, updating tracker
        region, updated_tracker, is_repeated = find_loop(junctions, start_edge, start_edge[1], lines_on, tracker)
        if region is None:
            region, updated_tracker, is_repeated = find_loop(junctions, start_edge, start_edge[0], lines_on, tracker)
        tracker = updated_tracker
        if not is_repeated:
            regions.append(region)

        # find next edge that does not appear twice in tracker
        start_edge = None
        for (i, j) in lines_on:
            if ((i, j) not in tracker):
                start_edge = (i, j)
                tracker.add((i, j))
                break
            if ((j, i) not in tracker):
                start_edge = (j, i)
                tracker.add((j, i))
                break

        # else stop
        if start_edge is None:
            break

    # mark regions using edges with intersection
    false_pos = []
    for l, poly in enumerate(regions):
        for k in range(len(poly)-1):
            i, j = poly[k], poly[k+1]
            if ((i, j) in edges_intersect) or ((j, i) in edges_intersect):
               false_pos.append(l)
               break

    polygons = []
    for region in regions:
        polygons.append([tuple(junctions[x]) for x in region])
    return polygons, false_pos


def fill_regions(edge_mask):
    edge_mask = edge_mask
    tag = 2
    for i in range(edge_mask.shape[0]):
        for j in range(edge_mask.shape[1]):
            if edge_mask[i, j] == 0:
                edge_mask = _flood_fill(edge_mask, i, j, tag)
                tag += 1
    return edge_mask

def _flood_fill(edge_mask, x0, y0, tag):
    new_edge_mask = np.array(edge_mask)
    nodes = [(x0, y0)]
    new_edge_mask[x0, y0] = tag
    while len(nodes) > 0:
        x, y = nodes.pop(0)
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (0 <= x+dx < new_edge_mask.shape[0]) and (0 <= y+dy < new_edge_mask.shape[0]) and (new_edge_mask[x+dx, y+dy] == 0):
                new_edge_mask[x+dx, y+dy] = tag
                nodes.append((x+dx, y+dy))
    return new_edge_mask


if __name__ == '__main__':

    # DEBUG
    junctions = np.array([(50, 50), (50, 100), (100, 100), (100, 50), (75, 75), (80, 80), (82, 81), (90, 93)])
    juncs_on = np.array([1, 1, 1, 1])
    lines_on = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 1), (4, 5), (5, 6), (6, 7)])
    extract_regions_v2(junctions, juncs_on, lines_on)
