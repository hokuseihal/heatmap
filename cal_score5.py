import numpy as np
import csv
from core import readxml, classname
from cal_score3 import cal_iou
import pickle
def precision_recall(csvfilename='detect_ssd_inception_test.csv', oklist=None, iou_thresh=.5, prob_thresh=.5,test_prob_yolo=0.5,test_prob_out=.5,test_prob_cut=0.5,ret=False):
    detect_dic = {}
    tp = np.zeros(len(classname))
    fp = np.zeros(len(classname))
    tpfn = np.zeros(len(classname))
    with open(csvfilename,'r') as f:
        reader = csv.reader((line.replace('\0','') for line in f) )
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
    if ret:
        return {'precision':precision, 'recall'   :recall, 'f_value'  :(2 / (1 / precision + 1 / recall)), 'mean':(np.mean(2 / (1 / precision + 1 / recall))),'yolo':test_prob_yolo,'cut':test_prob_cut,'out':test_prob_out}

def tester(testcsv,oklist,probthresh,resultlist):
    mx={'mean':0}
    for yolo in np.linspace(0.1,0.9,9):
        for out in np.linspace(0.1,0.9,9):
            for cut in np.linspace(0.1,0.9,9):
                now=precision_recall(testcsv, oklist, prob_thresh=probthresh, test_prob_yolo=yolo, test_prob_out=out,
                                     test_prob_cut=cut,ret=True)
                if resultlist[int(yolo*10)][int(out*10)][int(cut*10)]['mean']<now['mean']:
                    resultlist[int(yolo * 10)][int(out * 10)][int(cut * 10)]=now
                    print(f'yolo:{yolo},out:{out},cut:{cut}')
                    print(now)
                if mx['mean']<now['mean']:
                    mx=now


    print(mx)
    with open('resultlist.pkl','wb') as resultlist_f:
        pickle.dump(resultlist,resultlist_f)
    return mx
import csv
def showresult(resultlistpkl):
    with open(resultlistpkl,'rb') as f:
        resultlist=pickle.load(f)
    resultlist=[i for k in resultlist for j in k for i in j]
    resultlist=sorted(resultlist,key=lambda x:x['mean'])
    #print(resultlist)


class Precison_Recall_Teseter:
    def __init__(self,testcsv,probthresh):
        self.testcsv=testcsv
        self.probthresh=probthresh
        self.resultlist=[[[{'mean':0} for i in range(10)] for j in range(10)] for k in range(10)]
        self.maxresult={'mean':0}
    def precisoin_recall_test(self,oklist):
        maxinepoch=tester(testcsv=self.testcsv,oklist=oklist,probthresh=self.probthresh,resultlist=self.resultlist)
        if self.maxresult['mean']<maxinepoch['mean']:
            self.maxresult=maxinepoch
        print(self.resultlist)
        print('')

if __name__ == '__main__':
    precision_recall()
    #showresult('resultlist.pkl')

