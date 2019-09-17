from __future__ import print_function
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

class MAPMetric():

    def __init__(self,
                 class_names):

        self.class_names = class_names
        self.IOUs_all = []
        self.scores_all = []
        self.number_groundtruth_all = 0
        #self.number_prediction_all = 0

    def update(self,
               bboxes_groundtruth,
               labels_groundtruth,
               bboxes_prediction,
               labels_prediction,
               scores_prediction):

        bboxes_prediction  = np.array(bboxes_prediction,  dtype=np.float16)
        scores_prediction  = np.array(scores_prediction,  dtype=np.float16)
        bboxes_groundtruth = np.array(bboxes_groundtruth, dtype=np.float16)

        number_groundtruth = len(bboxes_groundtruth)
        number_prediction  = len(bboxes_prediction)
        self.number_groundtruth_all += number_groundtruth
        #self.number_prediction_all  += number_prediction

        matrix_IOU = np.empty((number_groundtruth,
                               number_prediction),
                              dtype=np.float16)
        for i,label_groundtruth in enumerate(labels_groundtruth):
            for j,label_prediction in enumerate(labels_prediction):
                if label_groundtruth == label_prediction:
                    matrix_IOU[i,j] = self._get_IOU(bboxes_groundtruth[i],
                                                    bboxes_prediction[j])
                else:
                    matrix_IOU[i,j] = 0
        #print(matrix_IOU)

        self.matched = linear_assignment(-matrix_IOU)
        #print(self.matched)
        self.unmatched_groundtruth = list(set(range(number_groundtruth))-set(self.matched[:,0]))
        self.unmatched_prediction  = list(set(range(number_prediction ))-set(self.matched[:,1]))
        for i,j in self.matched:
            self.IOUs_all.append(matrix_IOU[i,j])
            self.scores_all.append(scores_prediction[j])
        for j in self.unmatched_prediction:
            self.IOUs_all.append(0)
            self.scores_all.append(scores_prediction[j])

    @staticmethod
    def _get_IOU(bbox1,bbox2):
        x_overlap = np.float32( min(bbox1[2],bbox2[2]) - max(bbox1[0],bbox2[0]) )
        y_overlap = np.float32( min(bbox1[3],bbox2[3]) - max(bbox1[1],bbox2[1]) )
        if x_overlap<=0. or y_overlap<=0.:
            return 0.
        area1 = np.float32(bbox1[2]-bbox1[0])*np.float32(bbox1[3]-bbox1[1])
        area2 = np.float32(bbox2[2]-bbox2[0])*np.float32(bbox2[3]-bbox2[1])
        area_overlap = x_overlap*y_overlap
        return area_overlap/(area1+area2-area_overlap)

    def get_result(self,
                   type_mAP,
                   _threshes_IOU=[],
                   _threshes_recal=[]):

        if type_mAP == "VOC":
            threshes_IOU = [0.5]
            threshes_recall = []
            #threshes_recall = np.arange(0.00,1.01,0.10)
        elif type_mAP == "COCO":
            threshes_IOU = np.arange(0.50,1.00,0.05)
            threshes_recall = np.arange(0.00,1.01,0.01)
        elif type_mAP == "USER_DEFINED":
            threshes_IOU = _threshes_IOU
            threshes_recall = _threshes_recall
        self.mAP = 0.
        for thresh_IOU in threshes_IOU:
            area = self.get_area(thresh_IOU,threshes_recall)
            self.mAP += area/len(threshes_IOU)

    def get_area(self,thresh_IOU,threshes_recall):
        self.get_precision_recall_curve(thresh_IOU)
        area = 0.
        if len(threshes_recall) == 0: # calculate area under curve
            indices = np.where(self.recalls[1:]!=self.recalls[:-1])[0]
            area = np.sum( self.precisions[indices+1] * (self.recalls[indices+1]-self.recalls[indices]) )
        else:
            #for thresh_recall in sorted(threshes_recall):
            #    index = np.searchsorted(self.recalls,thresh_recall,side="left")
            #    precision = self.precisions[index]
            #    area += precision/len(threshes_recall)
            indices = np.searchsorted(self.recalls,threshes_recall,side="left")
            area = np.mean(self.precisions[indices])
        return area

    def get_precision_recall_curve(self,thresh_IOU):
        self.IOUs_all, self.scores_all = zip(*sorted(zip(self.IOUs_all,
                                                         self.scores_all),
                                                     key=lambda x:x[1]+x[0]*1e-10, reverse=True))
        #[print("%7.5f    %7.5f"%(self.IOUs_all[i],self.scores_all[i])) for i in range(len(self.IOUs_all))]
        TP = 0. ; FP = 0.
        precisions = [] ; recalls = []
        for IOU in self.IOUs_all:
            if IOU > thresh_IOU:
                TP += 1.
            else:
                FP += 1.
            if TP+FP > 0:
                recall = TP/self.number_groundtruth_all
                precision = TP/(TP+FP)
                recalls.append(recall)
                precisions.append(precision)
        for i in range(len(precisions)-1,0,-1):
            precisions[i-1] = max(precisions[i-1],precisions[i])
        precisions.append(0.)
        recalls.append(1.)
        self.precisions, self.recalls = zip(*sorted(list(set(zip(precisions,recalls))),
                                                    key=lambda x:x[1]-x[0]/(2*self.number_groundtruth_all)))
        self.precisions = np.array(self.precisions)
        self.recalls    = np.array(   self.recalls)

if __name__ == "__main__":
    import time

    bboxes_prediction  = [[000,100,100,200],
                          [000,100,100,201]]
    labels_prediction  = [0,0]
    scores_prediction  = [20,30]
    bboxes_groundtruth = [[010,110,110,210]]
#                          [000,100,100,201]]
    labels_groundtruth = [0]

    metric = MAPMetric("")
    metric.update(bboxes_groundtruth = bboxes_groundtruth,
                  labels_groundtruth = labels_groundtruth,
                  bboxes_prediction  = bboxes_prediction,
                  labels_prediction  = labels_prediction,
                  scores_prediction  = scores_prediction)

    metric.get_result(type_mAP="VOC")
    print(metric.mAP)
