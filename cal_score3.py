import csv
import xml.etree.ElementTree as ET
import numpy as np

filename = 'catyolo.csv'
dataroot = 'All/Annotations'
classes = ['D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44', 'D30']
resultmat = np.zeros((len(classes), len(classes)), dtype=int)
realclslist = np.zeros(len(classes), dtype=int)
preclslist = np.zeros(len(classes), dtype=int)
seenlist = []
fp = 0
n = 0


def cal_iou(r, x):
    global n
    r[1] = int(r[1])
    r[3] = int(r[3])
    r[4] = int(r[4])
    r[5] = int(r[5])
    r[6] = int(r[6])
    if not (r[3] < x[2] and x[0] < r[5] and r[4] < x[3] and x[1] < r[6]):
        return -1

    A = (x[2] - x[0]) * (x[3] - x[1])
    B = (r[5] - r[3]) * (r[6] - r[4])
    C = (min(r[5], x[2]) - max(r[3], x[0])) * (min(r[6], x[3]) - max(r[4], x[1]))
    iou = C / (A + B - C)
    return iou


def cal(mat):
    global realclslist
    diag = mat.diagonal()
    prediction = diag / preclslist
    recall = diag / realclslist
    f1 = 2 * (recall + prediction) / (recall + prediction)
    accuracy = diag.sum() / realclslist.sum()
    print('prediction', prediction)
    print('recall', recall)
    print('accuracy', accuracy)
    # print('f1',f1)
    return prediction, recall, f1


def readxml(r):
    global readclslist
    r_list = []
    try:
        tree = ET.parse(dataroot + '/' + r[0] + '.xml')

        root = tree.getroot()
        for child in root.iter('object'):
            for object in child:
                if 'name' == object.tag:
                    cls = object.text

                    if not r[0] in seenlist:
                        realclslist[classes.index(cls)] += 1
                    else:
                        pass
                if 'bndbox' == object.tag:
                    r_list.append([int(xy.text) for xy in object] + [classes.index(cls)])
        if not r[0] in seenlist: seenlist.append(r[0])
        return r_list

    except FileNotFoundError:
        print('Oops! ' + dataroot + '/' + r[0] + '.xml is not Found')
        return []


def addresult(r, x, mat):
    r_index = r[1]
    x_index = x[4]
    mat[x_index, r_index] += 1


def precision_recall():
    with open(filename) as f:
        reader = csv.reader(f)

        for row in reader:
            preclslist[int(row[1])] += 1
            for xml in readxml(row):
                if cal_iou(row, xml) > 0.1:
                    addresult(row, xml, resultmat)

    cal(resultmat)


if __name__ == '__main__':
    precision_recall()
