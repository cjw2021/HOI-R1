# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QAHOI
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import numpy as np
import os
import gzip
import json
from tqdm import tqdm

import torch

from loguru import logger

import argparse


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint64):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def write(file_name, data, mode='wb'):
    with open(file_name, mode) as f:
        f.write(data)

def dump_json_object(dump_object, file_name, compress=False, indent=4):
        data = json.dumps(
            dump_object, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=indent)
        if compress:
            write(file_name, gzip.compress(data.encode('utf8')))
        else:
            write(file_name, data, 'w')

def compute_area(bbox, invalid=None):
    x1, y1, x2, y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area

def compute_iou(bbox1, bbox2, verbose=False):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)

    area1 = compute_area(bbox1, invalid=0.0)
    area2 = compute_area(bbox2, invalid=0.0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou

def match_hoi(pred_det, gt_dets):
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i, gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                del remaining_gt_dets[i]
                break

    return is_match, remaining_gt_dets

def compute_ap(precision, recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall >= t]
        if selected_p.size == 0:
            p = 0
        else:
            p = np.max(selected_p)
        ap += p / 11.

    return ap

def compute_pr(y_true, y_score, npos):
    sorted_y_true = [y for y, _ in
                    sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)

    if len(tp) == 0:
        return 0, 0, False

    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall, True

def compute_center_distacne(bbox1, bbox2, img_h, img_w):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2

    c_x1 = (x11 + x12) / 2.0 / img_w
    c_x2 = (x21 + x22) / 2.0 / img_w
    c_y1 = (y11 + y12) / 2.0 / img_h
    c_y2 = (y21 + y22) / 2.0 / img_h

    diff_x = c_x1 - c_x2
    diff_y = c_y1 - c_y2

    distance = np.linalg.norm(np.array([diff_x, diff_y]))
    return distance

def compute_large_area(bbox1, bbox2, img_h, img_w, invalid=0.0):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2

    if (x12 <= x11) or (y12 <= y11):
        area1 = invalid
    else:
        area1 = (x12 - x11 + 1) / img_w * (y12 - y11 + 1) / img_h
    
    if (x22 <= x21) or (y22 <= y21):
        area2 = invalid
    else:
        area2 = (x22 - x21 + 1) / img_w * (y22 - y21 + 1) / img_h
    
    area = max(area1, area2)

    return area


class HICOEvaluator():
    def __init__(self, preds, gts, dataset_path, out_dir, epoch, bins_num=10, use_nms=True, nms_thresh=0.5, nms_alpha=1, nms_beta=0.5):
        self.nms_alpha = nms_alpha
        self.nms_beta = nms_beta
        self.out_dir = out_dir
        self.epoch = epoch
        self.bins_num = bins_num
        self.bins = np.linspace(0, 1.0, self.bins_num+1)
        self.compute_extra = {'distance': compute_center_distacne, 'area': compute_large_area}
        self.extra_keys = list(self.compute_extra.keys())
        self.ap_compute_set = {k: v for k, v in zip(self.extra_keys, [self._ap_compute_set() for i in range(len(self.extra_keys))])}
        self.img_size_info = {}
        self.img_folder = os.path.join(dataset_path, 'images/test2015')
        self.anno_path = os.path.join(dataset_path, "annotations")
        self.annotations = self.load_gt_dets()
        self.hoi_list = json.load(open(os.path.join(self.anno_path, 'hoi_list_new.json'), 'r'))
        self.file_name_to_obj_cat = json.load(open(os.path.join(self.anno_path, 'file_name_to_obj_cat.json'), "r"))
        self.nms_thresh = nms_thresh

        self.global_ids = self.annotations.keys()
        self.hoi_id_to_num = json.load(open(os.path.join(self.anno_path, 'hoi_id_to_num.json'), "r"))
        self.rare_id_json = [key for key, item in self.hoi_id_to_num.items() if item['rare']]
        
        self.correct_mat = np.load(os.path.join(self.anno_path, 'corre_hico.npy'))
        self.valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                              37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                              58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                              72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                              82, 84, 85, 86, 87, 88, 89, 90)
        self.valid_verb_ids = list(range(1, 118))

        self.pred_anno = {}
        self.preds_t = []
        self.use_nms = use_nms
        self.max_hois = 100
        # Using GGNet evaluation: https://github.com/SherlockHolmes221/GGNet/blob/main/src/lib/eval/hico_eval_de_ko.py
        print("convert preds...")
        for img_preds, img_gts in tqdm(zip(preds, gts), total=len(preds)):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': self.valid_obj_ids[label]} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']

            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([self.valid_obj_ids.index(bboxes[object_id]['category_id']) for object_id in object_ids])
                masks = self.correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': self.valid_verb_ids[category_id], 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            filename = img_gts["file_name"].split('.')[0]
            self.preds_t.append({
                'filename':filename,
                'predictions': bboxes,
                'hoi_prediction': hois
            })
        
        if self.use_nms:
            self.preds_t = self.triplet_nms_filter(self.preds_t)

        for preds_i in self.preds_t:

            # convert
            global_id = preds_i["filename"]
            self.pred_anno[global_id] = {}
            hois = preds_i["hoi_prediction"]
            bboxes = preds_i["predictions"]
            for hoi in hois:
                obj_id = bboxes[hoi['object_id']]['category_id']
                obj_bbox = bboxes[hoi['object_id']]['bbox']
                sub_bbox = bboxes[hoi['subject_id']]['bbox']
                score = hoi['score']
                if score < 0.001:
                    continue
                verb_id = hoi['category_id']

                hoi_id = '0'
                for item in self.hoi_list:
                    if item['object_cat'] == obj_id and item['verb_id'] == verb_id:
                        hoi_id = item['id']
                assert int(hoi_id) > 0

                data = np.array([sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                                obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3],
                                score]).reshape(1, 9)
                if hoi_id not in self.pred_anno[global_id]:
                    self.pred_anno[global_id][hoi_id] = np.empty([0, 9])

                self.pred_anno[global_id][hoi_id] = np.concatenate((self.pred_anno[global_id][hoi_id], data), axis=0)

    def load_gt_dets(self):
        # Load anno_list
        print('Loading anno_list.json ...')
        anno_list = json.load(open(os.path.join(self.anno_path, 'anno_list.json'), "r"))

        gt_dets = {}
        for anno in anno_list:
            if "test" not in anno['global_id']:
                continue

            global_id = anno['global_id']
            gt_dets[global_id] = {}
            img_h, img_w, _ = anno['image_size']
            self.img_size_info[global_id] = [img_h, img_w]
            for hoi in anno['hois']:
                hoi_id = hoi['id']
                gt_dets[global_id][hoi_id] = []
                for human_box_num, object_box_num in hoi['connections']:
                    human_box = hoi['human_bboxes'][human_box_num]
                    object_box = hoi['object_bboxes'][object_box_num]
                    det = {
                        'human_box': human_box,
                        'object_box': object_box,
                    }
                    gt_dets[global_id][hoi_id].append(det)

        return gt_dets

    def _ap_compute_set(self):
        out = {
            'y_true': [[] for i in range(self.bins_num)],
            'y_score': [[] for i in range(self.bins_num)],
            'npos': [0 for i in range(self.bins_num)]
        }
        return out

    def match_hoi_extra(self, pred_det, gt_dets, img_h, img_w):
        is_match = False
        remaining_gt_dets = [gt_det for gt_det in gt_dets]
        extra_info = {}
        for extra_i in self.extra_keys:
            extra_info[extra_i+'_pred'] = self.compute_extra[extra_i](pred_det['human_box'], pred_det['object_box'], img_h, img_w)
        for i, gt_det in enumerate(gt_dets):
            human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
            if human_iou > 0.5:
                object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
                if object_iou > 0.5:
                    is_match = True
                    del remaining_gt_dets[i]
                    for extra_i in self.extra_keys:
                        extra_info[extra_i+'_gt'] = self.compute_extra[extra_i](gt_det['human_box'], gt_det['object_box'], img_h, img_w)
                    break

        return is_match, remaining_gt_dets, extra_info

    def evaluation_default(self):
        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations, self.pred_anno, self.out_dir)
            outputs.append(o)

        mAP = {
            'AP': {},
            'mAP': 0,
            'invalid': 0,
            'mAP_rare': 0,
            'mAP_non_rare': 0,
        }
        map_ = 0
        map_rare = 0
        map_non_rare = 0
        count = 0
        count_rare = 0
        count_non_rare = 0
        for ap, hoi_id in outputs:
            mAP['AP'][hoi_id] = ap
            if not np.isnan(ap):
                count += 1
                map_ += ap
                if hoi_id in self.rare_id_json:
                    count_rare += 1
                    map_rare += ap
                else:
                    count_non_rare += 1
                    map_non_rare += ap

        mAP['mAP'] = map_ / count
        mAP['invalid'] = len(outputs) - count
        mAP['mAP_rare'] = map_rare / count_rare
        mAP['mAP_non_rare'] = map_non_rare / count_non_rare

        mAP_json = os.path.join(
            self.out_dir,
            f'epo_{self.epoch}_mAP_default.json')
        dump_json_object(mAP, mAP_json)

        print(f'APs have been saved to {self.out_dir}')
        return {"mAP_def": mAP['mAP'], "mAP_def_rare": mAP['mAP_rare'], "mAP_def_non_rare": mAP['mAP_non_rare']}

    def evaluation_ko(self):
        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations,
                              self.pred_anno, mode="ko",
                              obj_cate=hoi['object_cat'])
            outputs.append(o)

        mAP = {
            'AP': {},
            'mAP': 0,
            'invalid': 0,
            'mAP_rare': 0,
            'mAP_non_rare': 0,
        }
        map_ = 0
        map_rare = 0
        map_non_rare = 0
        count = 0
        count_rare = 0
        count_non_rare = 0
        for ap, hoi_id in outputs:
            mAP['AP'][hoi_id] = ap
            if not np.isnan(ap):
                count += 1
                map_ += ap
                if hoi_id in self.rare_id_json:
                    count_rare += 1
                    map_rare += ap
                else:
                    count_non_rare += 1
                    map_non_rare += ap

        mAP['mAP'] = map_ / count
        mAP['invalid'] = len(outputs) - count
        mAP['mAP_rare'] = map_rare / count_rare
        mAP['mAP_non_rare'] = map_non_rare / count_non_rare

        mAP_json = os.path.join(
            self.out_dir,
            f'epo_{self.epoch}_mAP_ko.json')
        dump_json_object(mAP, mAP_json)

        print(f'APs have been saved to {self.out_dir}')
        return {"mAP_ko": mAP['mAP'], "mAP_ko_rare": mAP['mAP_rare'], "mAP_ko_non_rare": mAP['mAP_non_rare']}

    def evaluation_extra(self):
        for hoi in self.hoi_list:
            self.eval_extra(hoi['id'], self.global_ids, self.annotations, self.pred_anno)

        extra_AP = {k: v for k, v in zip(self.extra_keys, [[{} for j in range(self.bins_num)] for i in range(len(self.extra_keys))])}
        for bins_i in range(self.bins_num):
            for extra_i in self.extra_keys:
                y_true = self.ap_compute_set[extra_i]['y_true'][bins_i]
                y_score = self.ap_compute_set[extra_i]['y_score'][bins_i]
                y_npos = self.ap_compute_set[extra_i]['npos'][bins_i]

                # Compute PR
                precision, recall, mark = compute_pr(y_true, y_score, y_npos)
                if not mark:
                    ap = 0
                else:
                    ap = compute_ap(precision, recall)
                if not np.isnan(ap):
                    extra_AP[extra_i][bins_i] = [ap, y_npos]
                else:
                    extra_AP[extra_i][bins_i] = [-1.0, y_npos]

        extra_AP_json = os.path.join(
            self.out_dir,
            f'epo_{self.epoch}_mAP_extra.json')
        dump_json_object(extra_AP, extra_AP_json)

        print(f'APs have been saved to {self.out_dir}')
        return {"extra_AP": extra_AP}

    def eval_hoi(self, hoi_id, global_ids, gt_dets, pred_anno,
                 mode='default', obj_cate=None):
        y_true = []
        y_score = []
        det_id = []
        npos = 0
        for global_id in global_ids:
            if mode == 'ko':
                if global_id + ".jpg" not in self.file_name_to_obj_cat:
                    continue
                obj_cats = self.file_name_to_obj_cat[global_id + ".jpg"]
                if int(obj_cate) not in obj_cats:
                    continue
            
            if hoi_id in gt_dets[global_id]:
                candidate_gt_dets = gt_dets[global_id][hoi_id]
            else:
                candidate_gt_dets = []

            npos += len(candidate_gt_dets)

            if global_id not in pred_anno or hoi_id not in pred_anno[global_id]:
                hoi_dets = np.empty([0, 9])
            else:
                hoi_dets = pred_anno[global_id][hoi_id]

            num_dets = hoi_dets.shape[0]

            sorted_idx = [idx for idx, _ in sorted(
                zip(range(num_dets), hoi_dets[:, 8].tolist()),
                key=lambda x: x[1],
                reverse=True)]
            for i in sorted_idx:
                pred_det = {
                    'human_box': hoi_dets[i, :4],
                    'object_box': hoi_dets[i, 4:8],
                    'score': hoi_dets[i, 8]
                }
                is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
                y_true.append(is_match)
                y_score.append(pred_det['score'])
                det_id.append((global_id, i))

        # Compute PR
        precision, recall, mark = compute_pr(y_true, y_score, npos)
        if not mark:
            ap = 0
        else:
            ap = compute_ap(precision, recall)
        # Compute AP
        return (ap, hoi_id)

    def eval_extra(self, hoi_id, global_ids, gt_dets, pred_anno):
        npos_all = 0
        npos_extra_all = {k: 0 for k in self.extra_keys}
        for global_id in global_ids:
            img_h, img_w = self.img_size_info[global_id]
            if hoi_id in gt_dets[global_id]:
                candidate_gt_dets = gt_dets[global_id][hoi_id]
            else:
                candidate_gt_dets = []
            npos_all += len(candidate_gt_dets)

            if global_id not in pred_anno or hoi_id not in pred_anno[global_id]:
                hoi_dets = np.empty([0, 9])
            else:
                hoi_dets = pred_anno[global_id][hoi_id]

            num_dets = hoi_dets.shape[0]

            sorted_idx = [idx for idx, _ in sorted(
                zip(range(num_dets), hoi_dets[:, 8].tolist()),
                key=lambda x: x[1],
                reverse=True)]
            for i in sorted_idx:
                pred_det = {
                    'human_box': hoi_dets[i, :4],
                    'object_box': hoi_dets[i, 4:8],
                    'score': hoi_dets[i, 8]
                }
                is_match, candidate_gt_dets, extra_info = self.match_hoi_extra(pred_det, candidate_gt_dets, img_h, img_w)
                for extra_i in self.extra_keys:
                    if is_match:
                        in_bins = np.min(np.argsort(np.abs(self.bins - extra_info[extra_i+'_gt']))[:2])
                        self.ap_compute_set[extra_i]['npos'][in_bins] += 1
                        npos_extra_all[extra_i] += 1
                    else:
                        in_bins = np.min(np.argsort(np.abs(self.bins - extra_info[extra_i+'_pred']))[:2])
                    self.ap_compute_set[extra_i]['y_true'][in_bins].append(is_match)
                    self.ap_compute_set[extra_i]['y_score'][in_bins].append(pred_det['score'])

            # add npos remain
            for remain_gt_det in candidate_gt_dets:
                for extra_i in self.extra_keys:
                    extra_gt = self.compute_extra[extra_i](remain_gt_det['human_box'], remain_gt_det['object_box'], img_h, img_w)
                    in_bins = np.min(np.argsort(np.abs(self.bins - extra_gt))[:2])
                    self.ap_compute_set[extra_i]['npos'][in_bins] += 1
                    npos_extra_all[extra_i] += 1
        
        for extra_i in self.extra_keys:
            assert npos_extra_all[extra_i] == npos_all

    # Refer to CDN: https://github.com/YueLiao/CDN/blob/main/datasets/hico_eval.py
    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in tqdm(preds):
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                          str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                'filename': img_preds['filename'],
                'predictions': pred_bboxes,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
            })

        return preds_filtered

    # Modified from CDN: https://github.com/YueLiao/CDN/blob/main/datasets/hico_eval.py
    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter/sub_union, self.nms_alpha) * np.power(obj_inter/obj_union, self.nms_beta)
            inds = np.where(ovr <= self.nms_thresh)[0]

            order = order[inds + 1]
        return keep_inds


def convert_hoi_predictions(preds, subject_category_id, object_class_to_idx, verb_class_to_idx):
    """
    Convert HOI JSON predictions to evaluator format, filtering out any interactions
    whose object or verb classes are not present in the provided mappings.

    Args:
        preds (dict): Dictionary containing HOI predictions.
        subject_category_id (int): Numeric ID for the subject class (e.g., human).
        object_class_to_idx (dict): Mapping from object class names to numeric IDs.
        verb_class_to_idx (dict): Mapping from verb class names to numeric IDs.

    Returns:
        list of dict: Each dict contains keys 'labels', 'boxes', 'verb_scores', 'sub_ids', 'obj_ids'.
    """

    results = []
    num_verb_classes = len(verb_class_to_idx)

    for interactions in preds:
        # Filter out interactions with invalid object or verb classes
        valid_interactions = []
        if not interactions is None:
            for inter in interactions:
                if not isinstance(inter, dict):
                    continue
                if 'object class' not in inter or 'verb class' not in inter:
                    continue
                if isinstance(inter['object class'], list):
                    if len(inter['object class']) == 0:
                        continue
                    inter['object class'] = inter['object class'][0]
                if "human" in inter and "object" in inter:
                    if not isinstance(inter['human'], list) or not isinstance(inter['object'], list):
                        continue
                    if len(inter['human']) == 4 and len(inter['object']) == 4:
                        if isinstance(inter['verb class'], list):
                            for v in inter['verb class']:
                                if inter['object class'] in object_class_to_idx and v in verb_class_to_idx:
                                    inter_t = inter.copy()
                                    inter_t['verb class'] = v
                                    valid_interactions.append(inter_t)
                        elif isinstance(inter['verb class'], str):
                            if ", " in inter['verb class']:
                                for v in inter['verb class'].split(", "):
                                    if inter['object class'] in object_class_to_idx and v in verb_class_to_idx:
                                        inter_t = inter.copy()
                                        inter_t['verb class'] = v
                                        valid_interactions.append(inter_t)
                            else:
                                if inter['object class'] in object_class_to_idx and inter['verb class'] in verb_class_to_idx:
                                    valid_interactions.append(inter)
        if len(valid_interactions) == 0:
            labels = torch.tensor(np.empty([0, 80]), dtype=torch.long)
            boxes = torch.tensor(np.empty([0, 4]), dtype=torch.float32)
            verb_scores = torch.tensor(np.empty([0, 117]), dtype=torch.float32)
            sub_ids = torch.tensor(np.empty([0]), dtype=torch.long)
            obj_ids = torch.tensor(np.empty([0]), dtype=torch.long)
            results.append({
                'labels': labels,
                'boxes': boxes,
                'verb_scores': verb_scores,
                'sub_ids': sub_ids,
                'obj_ids': obj_ids
            })
            continue
        interactions = valid_interactions
        nq = len(interactions)

        # Subject labels (all the same)
        subj_labels = torch.full((nq,), subject_category_id, dtype=torch.long)
        # Object labels from mapping
        obj_labels = torch.tensor([
            object_class_to_idx[inter['object class']] for inter in interactions
        ], dtype=torch.long)
        labels = torch.cat((subj_labels, obj_labels), dim=0).cpu()

        # Subject and object boxes
        subj_boxes = torch.tensor([
            inter['human'] for inter in interactions
        ], dtype=torch.float32)
        obj_boxes = torch.tensor([
            inter['object'] for inter in interactions
        ], dtype=torch.float32)
        boxes = torch.cat((subj_boxes, obj_boxes), dim=0).cpu()

        # Verb scores: linear from 1.0 (first) to 0.5 (last)
        if nq > 1:
            scores = [1.0 - i * 0.5 / (nq - 1) for i in range(nq)]
        else:
            scores = [1.0]
        verb_scores = torch.zeros((nq, num_verb_classes), dtype=torch.float32)
        for i, inter in enumerate(interactions):
            vid = verb_class_to_idx[inter['verb class']]
            verb_scores[i, vid] = scores[i]
        verb_scores = verb_scores.cpu()

        # Subject and object IDs
        ids = torch.arange(2 * nq, dtype=torch.long)
        sub_ids = ids[:nq]
        obj_ids = ids[nq:]

        results.append({
            'labels': labels,
            'boxes': boxes,
            'verb_scores': verb_scores,
            'sub_ids': sub_ids,
            'obj_ids': obj_ids
        })

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HICO-DET evaluation script')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the merged_output.json file containing predictions')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for evaluation results')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the HICO-DET root (contains images/test2015 and annotations)')
    args = parser.parse_args()

    json_path = args.json_path
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load the JSON predictions
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Example usage
    subject_category_id = 0
    ACT_IDX_TO_ACT_NAME = [
        'adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with',
        'buy', 'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook',
        'cut', 'cut_with', 'direct', 'drag', 'dribble', 'drink_with', 'drive',
        'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly',
        'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug',
        'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso', 'launch',
        'lick', 'lie_on', 'lift', 'light', 'load', 'lose', 'make', 'milk', 'move',
        'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay',
        'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race',
        'read', 'release', 'repair', 'ride', 'row', 'run', 'sail', 'scratch',
        'serve', 'set', 'shear', 'sign', 'sip', 'sit_at', 'sit_on', 'slide',
        'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick',
        'stir', 'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach',
        'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk',
        'wash', 'watch', 'wave', 'wear', 'wield', 'zip'
    ]

    OBJ_IDX_TO_OBJ_NAME = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite',
        'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
        'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant',
        'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
        'hair_drier', 'toothbrush'
    ]
    object_class_to_idx = {obj.replace("_", " "): i for i, obj in enumerate(OBJ_IDX_TO_OBJ_NAME)}
    verb_class_to_idx = {act.replace("_", " "): i for i, act in enumerate(ACT_IDX_TO_ACT_NAME)}
    results = convert_hoi_predictions(
        preds=data["preds"],
        subject_category_id=subject_category_id,
        object_class_to_idx=object_class_to_idx,
        verb_class_to_idx=verb_class_to_idx
    )
    
    evaluator = HICOEvaluator(results, data["gt_paths"], args.dataset_path, out_dir, 1, use_nms=False, nms_thresh=0.7, nms_alpha=1.0, nms_beta=0.5)

    stats = evaluator.evaluation_default()
    logger.info('\n--------------------\ndefault mAP: {}\ndefault mAP rare: {}\ndefault mAP non-rare: {}\n--------------------'.format(stats['mAP_def'], stats['mAP_def_rare'], stats['mAP_def_non_rare']))
    stats_ko = evaluator.evaluation_ko()
    logger.info('\n--------------------\nko mAP: {}\nko mAP rare: {}\nko mAP non-rare: {}\n--------------------'.format(stats_ko['mAP_ko'], stats_ko['mAP_ko_rare'], stats_ko['mAP_ko_non_rare']))
    stats.update(stats_ko)

