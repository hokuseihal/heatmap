import numpy as np
import csv
from core import readxml, classname
from cal_score3 import cal_iou


def precision_recall(csvfilename='y2rresult_001.csv',oklist=None, iou_thresh=.5):
    detect_dic = {}
    tp = np.zeros(len(classname))
    fp = np.zeros(len(classname))
    tpfn = np.zeros(len(classname))
    with open(csvfilename) as f:
        reader = csv.reader(f)
        for idx,csv_detect_row in enumerate(reader):
            if oklist is None or oklist[idx]:
                detect_dic.setdefault(csv_detect_row[0], []).append(csv_detect_row)
    for detectfilename in detect_dic:
        detections_on_img = detect_dic[detectfilename]
        detections_on_img_id = np.zeros(len(detections_on_img))
        detections_on_img_id_cls = np.zeros(len(detections_on_img), dtype=int)
        groundtruth_on_img = readxml(detectfilename)
        ground_truth_on_img_id = np.zeros(len(groundtruth_on_img))
        ground_truth_on_img_id_cls = np.zeros(len(groundtruth_on_img), dtype=int)
        for det_idx, det in enumerate(detections_on_img):
            for gtb_idx, gtb in enumerate(groundtruth_on_img):
                det_cls = int(det[1])
                gt_cls = gtb[-1]
                detections_on_img_id_cls[det_idx] = det_cls
                ground_truth_on_img_id_cls[gtb_idx] = gt_cls
                if cal_iou(det, gtb) > iou_thresh and det_cls == gt_cls:
                    detections_on_img_id[det_idx] += 1
                    ground_truth_on_img_id[gtb_idx] += 1
        # if map : add dets if id==1
        _tp = np.eye(len(classname))[detections_on_img_id_cls][detections_on_img_id == 1].sum(axis=0)
        _fp = np.eye(len(classname))[detections_on_img_id_cls][detections_on_img_id == 0].sum(axis=0)
        _tpfn = np.eye(len(classname))[ground_truth_on_img_id_cls].sum(axis=0)
        if len(_tp) != 0:
            tp += _tp.reshape(len(classname))
        if len(_fp) != 0:
            fp += _fp.reshape(len(classname))
        if len(_tpfn) != 0:
            tpfn += _tpfn.reshape(len(classname))
    # if map:cal map here #recall is tp-all(all)
    precision=(tp / (tp + fp))[:6]
    recall=(tp / (tpfn))[:6]
    print(f'precision:{precision}')
    print(f'recall:{recall}')
    print(f'f_value{1/(1/precision+1/recall)}')
    print(f'mean{np.mean(1/(1/precision+1/recall))}')


if __name__=='__main__':
    precision_recall()
