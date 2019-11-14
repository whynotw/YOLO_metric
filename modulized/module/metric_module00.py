from __future__ import print_function
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

class ObjectDetectionMetric():

    def __init__(self,
                 names_class,
                 check_class_first=True):

        self.check_class_first = check_class_first
        self.names_class = names_class
        self.number_classes = len(names_class)
        self.number_groundtruth_all = [0]*self.number_classes
        self.number_prediction_all  = [0]*self.number_classes
        self.infos_all = [ [] for _ in range(self.number_classes+1) ]
        self._is_sorted = False

    def update(self,
               bboxes_groundtruth,
               labels_groundtruth,
               bboxes_prediction,
               labels_prediction,
               scores_prediction):

        self._is_sorted = False

        bboxes_prediction  = np.array(bboxes_prediction,  dtype=np.float32)
        scores_prediction  = np.array(scores_prediction,  dtype=np.float32)
        labels_prediction  = np.array(labels_prediction,  dtype=np.uint16)
        bboxes_groundtruth = np.array(bboxes_groundtruth, dtype=np.float32)
        labels_groundtruth = np.array(labels_groundtruth, dtype=np.uint16)

        number_groundtruth = len(bboxes_groundtruth)
        number_prediction  = len(bboxes_prediction)
        for label in labels_groundtruth:
            self.number_groundtruth_all[label] += 1
        for label in labels_prediction:
            self.number_prediction_all[label] += 1

        matrix_IOU = self._get_IOUs(bboxes_groundtruth,bboxes_prediction)
        same = labels_groundtruth.reshape((-1,1))==labels_prediction.reshape((1,-1))
        if self.check_class_first:
            matrix_IOU *= same
        else:
            matrix_IOU[same] *= (1+1e-06)
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
                self.infos_all[ labels_prediction[j] ].append([labels_groundtruth[i],
                                                               scores_prediction[j],
                                                               matrix_IOU[i,j]])

        for i in self.unmatched_groundtruth:
            self.infos_all[ -1 ].append( [labels_groundtruth[i],0,0] )

        for j in self.unmatched_prediction:
            self.infos_all[ labels_prediction[j] ].append( [-1,scores_prediction[j],0] )

    @staticmethod
    def _get_IOUs(bboxes1_xywh,bboxes2_xywh):
        if bboxes1_xywh.shape[0] == 0:
            return np.zeros((len(bboxes1_xywh),len(bboxes2_xywh)))
        if bboxes2_xywh.shape[0] == 0:
            return np.zeros((len(bboxes1_xywh),len(bboxes2_xywh)))

        split = len(bboxes1_xywh)
        bboxes = np.concatenate( (bboxes1_xywh,bboxes2_xywh) , axis=0)
        centers = bboxes[:,0:2]
        sizes   = bboxes[:,2:4]
        bboxes = np.concatenate( (centers-sizes/2.,centers+sizes/2.) , axis=-1) # left, top, right, bottom
        bboxes1 = (bboxes[:split]).reshape((-1,1,4))
        bboxes2 = (bboxes[split:]).reshape((1,-1,4))
        x_overlap = np.minimum(bboxes1[:,:,2],bboxes2[:,:,2]) - np.maximum(bboxes1[:,:,0],bboxes2[:,:,0]) # right-left
        y_overlap = np.minimum(bboxes1[:,:,3],bboxes2[:,:,3]) - np.maximum(bboxes1[:,:,1],bboxes2[:,:,1]) # bottom-top
        area_i = np.where( (x_overlap>0) & (y_overlap>0) , x_overlap*y_overlap, 0.)
        area1 = ( bboxes1[:,:,2]-bboxes1[:,:,0] )*( bboxes1[:,:,3]-bboxes1[:,:,1] )
        area2 = ( bboxes2[:,:,2]-bboxes2[:,:,0] )*( bboxes2[:,:,3]-bboxes2[:,:,1] )
        return area_i/(area1+area2-area_i)

    def _sort(self):
        self.IOUs_all   = [ [] for _ in range(self.number_classes) ]
        self.scores_all = [ [] for _ in range(self.number_classes) ]
        for no_class in range(self.number_classes):
            infos = np.array( self.infos_all[no_class] ).reshape((-1,3))
            matched = ( infos[:,0] == no_class )
            unmatched = np.logical_not(matched)
            self.scores_all[no_class]  = list(infos[matched,1])
            self.IOUs_all[  no_class]  = list(infos[matched,2])
            self.scores_all[no_class] += list(infos[unmatched,1])
            self.IOUs_all[  no_class] += [0]*sum(unmatched)
            if len(self.IOUs_all[no_class]) != 0:
                self.IOUs_all[  no_class],\
                self.scores_all[no_class] = zip(*sorted(zip(self.IOUs_all[no_class],
                                                            self.scores_all[no_class]),
                                                        key=lambda x:x[1]+x[0]*1e-10, reverse=True))
                #for i in range(len(self.IOUs_all[no_class])):
                #    print("%7.5f    %7.5f"%(self.IOUs_all[no_class][i],self.scores_all[no_class][i]))
                #print("--")
        self._is_sorted = True

    def get_mAP(self,
                type_mAP,
                _threshes_IOU=[],
                _threshes_recall=[],
                conclude=False):

        if not self._is_sorted:
            self._sort()

        if type_mAP == "VOC07":
            threshes_IOU = [0.5] # only 0.5
            threshes_recall = np.arange(0.00,1.01,0.10) # 11-point interpolation
        elif type_mAP == "VOC12":
            threshes_IOU = [0.5] # only 0.5
            threshes_recall = [] # area under curve
        elif type_mAP == "COCO":
            threshes_IOU = np.arange(0.50,1.00,0.05) # from 0.5 to 0.95
            threshes_recall = np.arange(0.00,1.01,0.01) # 101-point interpolation
        elif type_mAP == "USER_DEFINED":
            threshes_IOU = _threshes_IOU
            threshes_recall = _threshes_recall
        APs = [0.]*self.number_classes
        for no_class in range(self.number_classes):
            for thresh_IOU in threshes_IOU:
                area = self.get_area(no_class,thresh_IOU,threshes_recall)
                APs[no_class] += area/len(threshes_IOU)
        mAP = np.mean(APs)
        number_groundtruth_total = sum(self.number_groundtruth_all)+1e-10
        weighted_mAP = np.dot(APs,self.number_groundtruth_all) / number_groundtruth_total
        if not conclude:
           return # TODO return what you want
        # print the results
        length_name = max([len(str(name)) for name in self.names_class]+[8])
        length_number = len(str(number_groundtruth_total))
        spacing = "- "*(int(20+(length_name+length_number)/2))
        print(spacing+"\nMean Average Precision\n"+spacing)
        print("metric: %s"%type_mAP)
        for no_class in range(self.number_classes):
            print("[%*s]      AP: %6.2f %%   #: %*d (%6.2f %%)"%\
                  (length_name,self.names_class[no_class],
                   APs[no_class]*1e2,
                   length_number,self.number_groundtruth_all[no_class],
                   self.number_groundtruth_all[no_class]*1e2/number_groundtruth_total))
        print("[%*s]     mAP: %6.2f %%   #: %*d (100.00 %%)"%\
              (length_name,"total",
               mAP*1e2,
               length_number,number_groundtruth_total))
        print("[%*s]     mAP: %6.2f %%"%\
              (length_name,"weighted",
               weighted_mAP*1e2))
        print(spacing)

    def get_area(self,no_class,thresh_IOU,threshes_recall):
        if len(self.IOUs_all[no_class]) == 0:
            return 0. # no data in this class
        precisions, recalls = self.get_precision_recall_curve(no_class,thresh_IOU)
        if len(threshes_recall) == 0: # calculate area under curve
            indices = np.where(recalls[1:]!=recalls[:-1])[0]
            area = np.sum( precisions[indices+1]
                         * (recalls[indices+1]-recalls[indices]) )
            #area = np.trapz(precisions,x=recalls)
        else:
            indices = np.searchsorted(recalls,threshes_recall,side="left")
            area = np.mean(precisions[indices])
        return area

    def get_precision_recall_curve(self,no_class,thresh_IOU):
        hits = np.array(self.IOUs_all[no_class]) >= thresh_IOU
        TPs = np.cumsum(hits,dtype=np.float32)
        # precision function is decreasing when recall doesn't change, and increasing when recall changes
        # recall function is always increasing
        precisions = [0.] + list(TPs/np.arange(1,len(TPs)+1))               + [0.] # should be decreasing
        recalls    = [0.] + list(TPs/self.number_groundtruth_all[no_class]) + [1.] # always increasing
        for i in range(len(precisions)-1,0,-1): # recall value from large to small
            precisions[i-1] = max(precisions[i-1],precisions[i]) # precision should be larger and larger
            if recalls[i-1] == recalls[i]: # remove duplicate recall and keep the one with the largest precision
               precisions.pop(i)
               recalls.pop(i)
        return np.array(precisions), np.array(recalls)

    def get_confusion(self,
                      thresh_confidence,
                      thresh_IOU,
                      conclude=False):

        eps = 1e-50
        IOUs_avg = []
        matrix_confusion = np.zeros((self.number_classes+1,self.number_classes+1))
        for no_class_prediction in range(self.number_classes):
            infos = np.array(self.infos_all[no_class_prediction])
            infos = infos.reshape((-1,3))
            above_IOU = ( infos[:,2]>=thresh_IOU )
            same = ( infos[:,0]==no_class_prediction )
            IOUs_avg.append( np.sum(infos[:,2]*same)/(np.sum(same)+eps) ) # if matched
            below_IOU = np.logical_not(above_IOU)

            matrix_confusion[-1,no_class_prediction] += np.sum(below_IOU)
            for no_class_groundtruth in range(self.number_classes):
                matched_class = ( infos[:,0] == no_class_groundtruth )
                matrix_confusion[no_class_groundtruth,no_class_prediction] += np.sum(matched_class & above_IOU)
                matrix_confusion[no_class_groundtruth,-1]                  += np.sum(matched_class & below_IOU)

        infos = np.array(self.infos_all[-1])
        if len(infos):
            for no_class_groundtruth in range(self.number_classes):
                matrix_confusion[no_class_groundtruth,-1] += np.sum( infos[:,0]==no_class_groundtruth )

        if not conclude:
           return # TODO return what you want
        # print the results
        fields = self.names_class+["none"]
        length_name = max([len(str(s)) for s in fields]+[5])
        spacing = "- "*max((int(7+((length_name+3)*(self.number_classes+3))/2)),
                           length_name+33)
        print(spacing+"\nConfusion Matrix\n"+spacing)
        print(("thresh_confidence: %f"%thresh_confidence).rstrip("0"))
        print(("thresh_IOU       : %f"%thresh_IOU).rstrip("0"))
        matrix_confusion = np.uint32(matrix_confusion)
        content = " "*(length_name+3+12)
        for j in range(self.number_classes+1):
            content += "[%*s] "%(length_name,fields[j])
        content += "[%*s] "%(length_name,"total")
        print("%*sPrediction"%(12+(len(content)-10)/2,""))
        print(content)
        for i in range(self.number_classes+1):
            content = "Groundtruth " if i==int((self.number_classes+1)/2) else " "*12
            content += "[%*s] "%(length_name,fields[i])
            for j in range(self.number_classes+1):
                if i==j==self.number_classes:
                    break
                content += "%*d "%(length_name+2,matrix_confusion[i,j])
            if i < self.number_classes:
                content += "%*d "%(length_name+2,self.number_groundtruth_all[i])
            print(content)
        content = " "*12+"[%*s] "%(length_name,"total")
        for j in range(self.number_classes):
            content += "%*d "%(length_name+2,self.number_prediction_all[j])
            #content += "%*d "%(length_name+2,sum(matrix_confusion[:,j]))
        print(content)
        print(spacing)
        for no_class,name in enumerate(self.names_class):
            precision = matrix_confusion[no_class,no_class]/(self.number_prediction_all[ no_class]+eps)
            recall    = matrix_confusion[no_class,no_class]/(self.number_groundtruth_all[no_class]+eps)
            print(("[%*s]   precision: %6.2f %%"
                         "     recall: %6.2f %%"
                        "     avg IOU: %6.2f %%")%(\
                length_name,name,
                1e2*precision,
                1e2*recall,
                1e2*IOUs_avg[no_class]))
        print(spacing)

if __name__ == "__main__":

    bboxes_prediction  = [[000,100,100,200],
                          [000,100,100,201]]
    labels_prediction  = [0,0]
    scores_prediction  = [20,30]
    bboxes_groundtruth = [[010,110,110,210],
                          [000,100,100,201]]
    labels_groundtruth = [0,1]

    metric = MAPMetric([0,1,23])
    metric.update(bboxes_groundtruth = bboxes_groundtruth,
                  labels_groundtruth = labels_groundtruth,
                  bboxes_prediction  = bboxes_prediction,
                  labels_prediction  = labels_prediction,
                  scores_prediction  = scores_prediction)

    metric.get_mAP(type_mAP="VOC12",
                   conclude=True)
    precisions, recalls = metric.get_precision_recall_curve(no_class=0,thresh_IOU=0.5)
    #print(precisions)
    #print(recalls)
    metric.get_confusion(thresh_confidence=0.5,
                         thresh_IOU=0.5,
                         conclude=True)
