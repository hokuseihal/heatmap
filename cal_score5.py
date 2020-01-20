import numpy as np
import csv
from core import readxml, classname
from cal_score3 import cal_iou

def precision_recall(csvfilename='01test.csv', oklist=None, iou_thresh=.5, prob_thresh=.5,test_prob_yolo=0.5,test_prob_out=0.5,test_prob_cut=0,return_str=False):
    detect_dic = {}
    tp = np.zeros(len(classname))
    fp = np.zeros(len(classname))
    tpfn = np.zeros(len(classname))
    with open(csvfilename) as f:
        reader = csv.reader(f)
        lines=[i for i in reader if float(i[2])>prob_thresh]
        for idx, csv_detect_row in enumerate(lines):
            if (oklist is None or oklist[idx]>test_prob_out or float(csv_detect_row[2])>test_prob_yolo) and float(csv_detect_row[2])>test_prob_cut:
                detect_dic.setdefault(csv_detect_row[0], []).append(csv_detect_row)
            else:
                detect_dic.setdefault(csv_detect_row[0], [])
    for detectfilename in detect_dic:
        detections_on_img = detect_dic[detectfilename]
        detections_on_img_id = np.zeros(len(detections_on_img))
        detections_on_img_id_cls = np.zeros(len(detections_on_img), dtype=int)
        groundtruth_on_img = readxml(detectfilename)
        ground_truth_on_img_id = np.zeros(len(groundtruth_on_img))
        ground_truth_on_img_id_cls = np.zeros(len(groundtruth_on_img), dtype=int)
        for det_idx, det in enumerate(detections_on_img):
            if float(det[2]) < prob_thresh:
                detections_on_img_id[det_idx] = -1
                continue
            for gtb_idx, gtb in enumerate(groundtruth_on_img):

                det_cls = int(det[1])
                gt_cls = gtb[-1]
                detections_on_img_id_cls[det_idx] = det_cls
                ground_truth_on_img_id_cls[gtb_idx] = gt_cls
                if cal_iou(det, gtb) > iou_thresh and det_cls == gt_cls:
                    if ground_truth_on_img_id[gtb_idx] == 0:
                        detections_on_img_id[det_idx] = 1
                        ground_truth_on_img_id[gtb_idx] = 1
                    else:
                        detections_on_img_id[det_idx] = -1
                        # if map : add dets if id==1
        _tp = np.eye(len(classname))[detections_on_img_id_cls][detections_on_img_id == 1].sum(axis=0)
        _fp = np.eye(len(classname))[detections_on_img_id_cls][detections_on_img_id == 0].sum(axis=0)
        # _tpfn = np.eye(len(classname))[ground_truth_on_img_id_cls].sum(axis=0)
        if len(_tp) != 0:
            tp += _tp.reshape(len(classname))
        if len(_fp) != 0:
            fp += _fp.reshape(len(classname))
        # if len(_tpfn) != 0:
        #    tpfn += _tpfn.reshape(len(classname))
    valtxt = 'val.txt'
    with open(valtxt) as f:
        for xmlfile in f.readlines():
            bbx = readxml(xmlfile)
            for bb in bbx:
                tpfn[bb[-1]] += 1
    # if map:cal map here #recall is tp-all(all)

    precision = (tp / (tp + fp))[:6]
    recall = (tp / (tpfn))[:6]
    print(f'precision:{precision}')
    print(f'recall   :{recall}')
    print(f'f_value  :{2 / (1 / precision + 1 / recall)}')
    print(f'mean:{np.mean(2 / (1 / precision + 1 / recall))}')
    if return_str:
        return f'precision:{precision} recall   :{recall} f_value  :{2 / (1 / precision + 1 / recall)} :mean:{np.mean(2 / (1 / precision + 1 / recall))}\n'
    return np.mean(2 / (1 / precision + 1 / recall))

def tester(testcsv,oklist,probthresh):
    mx=0
    myolo=0
    mout=0
    mcut=0
    for yolo in np.linspace(0.1,0.9,9):
        for out in np.linspace(0.1,0.9,9):
            for cut in np.linspace(0.1,0.9,9):
                print(f'yolo:{yolo},out:{out},cut:{cut}')
                now=precision_recall(testcsv, oklist, prob_thresh=probthresh, test_prob_yolo=yolo, test_prob_out=out,
                                     test_prob_cut=cut)
                print(now)
                if mx<now:
                    mx=now
                    myolo=yolo
                    mout=out
                    mcut=cut
            if mx>0.515:
                with open('congrad.txt','a') as f:
                    f.write(f'yolo:{yolo},out:{out},cut:{cut}')
                    f.write(f'{precision_recall(testcsv, oklist, prob_thresh=probthresh, test_prob_yolo=yolo, test_prob_out=out,test_prob_cut=cut,return_str=True)}')
            if mx>0.52:
                print('!!!!!!!!!!!!!!!!CONGRATULATION!!!!!!!!!!!!!!!')
    print(f'max is {mx},yolo:{myolo},out:{mout},cut:{mcut}')
    precision_recall(csvfilename=testcsv,oklist=oklist,prob_thresh=probthresh,test_prob_yolo=myolo,test_prob_out=mout,test_prob_cut=mcut)

if __name__ == '__main__':
    precision_recall()

