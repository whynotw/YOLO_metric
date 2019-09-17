from __future__ import print_function
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

class MAPMetric():

    def __init__(self,
                 class_names):

        self.class_names = class_names
        self.number_classes = len(class_names)
        self.IOUs_all = []
        self.scores_all = []
        self.precisions = []
        self.recalls = []
        self.number_groundtruth_all = []
        for _ in range(self.number_classes):
            self.IOUs_all.append([])
            self.scores_all.append([])
            self.precisions.append([])
            self.recalls.append([])
            self.number_groundtruth_all.append(0)
            #self.number_prediction_all.append(0)

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
        for label in labels_groundtruth:
            self.number_groundtruth_all[label] += 1

        matrix_IOU = np.empty((number_groundtruth,
                               number_prediction),
                              dtype=np.float32)
        for i,label_groundtruth in enumerate(labels_groundtruth):
            for j,label_prediction in enumerate(labels_prediction):
                #if label_groundtruth == label_prediction:
                    matrix_IOU[i,j] = self._get_IOU(bboxes_groundtruth[i],
                                                    bboxes_prediction[j])\
                                    + (label_groundtruth==label_prediction)*1e-6
                #else:
                #    matrix_IOU[i,j] = 0
        #print(matrix_IOU)

        self.matched = linear_assignment(-matrix_IOU)
        self.unmatched_groundtruth = list(set(range(number_groundtruth))-set(self.matched[:,0]))
        self.unmatched_prediction  = list(set(range(number_prediction ))-set(self.matched[:,1]))
        for n,(i,j) in reversed(list(enumerate(self.matched))):
            if matrix_IOU[i,j] == 0:
                self.unmatched_groundtruth.append(i)
                self.unmatched_prediction.append(j)
                self.matched = np.delete(self.matched,n,0)
            else:
                if labels_groundtruth[i]==labels_prediction[j]:
                    self.IOUs_all[labels_prediction[j]].append(matrix_IOU[i,j])
                else:
                    self.IOUs_all[labels_prediction[j]].append(0.)
                self.scores_all[labels_prediction[j]].append(scores_prediction[j])

        for j in self.unmatched_prediction:
            self.IOUs_all[labels_prediction[j]].append(0)
            self.scores_all[labels_prediction[j]].append(scores_prediction[j])

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
        self.APs = [0.]*self.number_classes
        for no_class in range(self.number_classes):
            for thresh_IOU in threshes_IOU:
                area = self.get_area(no_class,thresh_IOU,threshes_recall)
                self.APs[no_class] += area/len(threshes_IOU)
        self.mAP = np.mean(self.APs)

    def conclude(self):
        number_groundtruth_total = sum(self.number_groundtruth_all)
        length_name = max([len(str(name)) for name in self.class_names])
        length_number = len(str(number_groundtruth_total))
        for no_class in range(self.number_classes):
            print("class name: %*s    AP: %6.2f %%   #: %*d (%6.2f %%)"%\
                  (length_name,self.class_names[no_class],
                   self.APs[no_class]*1e2,
                   length_number,
                   self.number_groundtruth_all[no_class],
                   self.number_groundtruth_all[no_class]*1e2/number_groundtruth_total))
        print("     total: %*s   mAP: %6.2f %%   #: %*d (100.00 %%)"%\
              (length_name,"",
               self.mAP*1e2,
               length_number,
               number_groundtruth_total))
        print("%*s weighted mAP: %6.2f %%"%\
              (length_name+5,"",
               np.sum(np.dot(self.APs,self.number_groundtruth_all))*1e2/number_groundtruth_total))

    def get_area(self,no_class,thresh_IOU,threshes_recall):
        self.precisions[no_class], self.recalls[no_class] = self.get_precision_recall_curve(no_class,thresh_IOU)
        area = 0.
        if len(threshes_recall) == 0: # calculate area under curve
            indices = np.where(self.recalls[no_class][1:]!=self.recalls[no_class][:-1])[0]
            area = np.sum( self.precisions[no_class][indices+1] * (self.recalls[no_class][indices+1]-self.recalls[no_class][indices]) )
        else:
            indices = np.searchsorted(self.recalls[no_class],threshes_recall,side="left")
            area = np.mean(self.precisions[no_class][indices])
        return area

    def get_precision_recall_curve(self,no_class,thresh_IOU):
        self.IOUs_all[no_class], self.scores_all[no_class] = zip(*sorted(zip(self.IOUs_all[no_class],
                                                                             self.scores_all[no_class]),
                                                                         key=lambda x:x[1]+x[0]*1e-10, reverse=True))
        #[print("%7.5f    %7.5f"%(self.IOUs_all[i],self.scores_all[i])) for i in range(len(self.IOUs_all))]
        TP = 0. ; FP = 0.
        precisions = [0.] ; recalls = [0.]
        for IOU in self.IOUs_all[no_class]:
            if IOU > thresh_IOU:
                TP += 1.
            else:
                FP += 1.
            if TP+FP > 0:
                recall = TP/self.number_groundtruth_all[no_class]
                precision = TP/(TP+FP)
                recalls.append(recall)
                precisions.append(precision)
        for i in range(len(precisions)-1,0,-1):
            precisions[i-1] = max(precisions[i-1],precisions[i])
        precisions.append(0.)
        recalls.append(1.)
        precisions, recalls = zip(*sorted(list(set(zip(precisions,recalls))),
                                                    key=lambda x:x[1]-x[0]/(2*self.number_groundtruth_all[no_class])))
        return np.array(precisions), np.array(recalls)

if __name__ == "__main__":
    import time

    bboxes_prediction  = [[000,100,100,200],
                          [000,100,100,201]]
    labels_prediction  = [0,0]
    scores_prediction  = [20,30]
    bboxes_groundtruth = [[010,110,110,210]]
#                          [000,100,100,201]]
    labels_groundtruth = [0]

    metric = MAPMetric([0])
    metric.update(bboxes_groundtruth = bboxes_groundtruth,
                  labels_groundtruth = labels_groundtruth,
                  bboxes_prediction  = bboxes_prediction,
                  labels_prediction  = labels_prediction,
                  scores_prediction  = scores_prediction)

    metric.get_result(type_mAP="VOC")
    metric.conclude()
    print(metric.precisions)
    print(metric.recalls)
