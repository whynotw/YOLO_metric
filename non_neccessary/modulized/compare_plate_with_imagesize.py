import numpy as np
import cv2
import os
import imagesize
from module import label_module, metric_module
#from module import metric_module_not_support_confusion_matrix_diff_thresh_IOU as metric_module
from module import metric_module_with_imagesize as metric_module

# settings

DIRNAME_TEST = "compare_plate/darknet_train_yolov3_tiny_car_plate_generated09_4000/test001"
DIRNAME_PREDICTION = os.path.join(DIRNAME_TEST,"labels_prediction")
FILENAME_GROUNDTRUTH = os.path.join(DIRNAME_TEST,"test.txt")

TEXTNAMES_PREDICTION = [ os.path.join(DIRNAME_PREDICTION,p) for p in sorted(os.listdir(DIRNAME_PREDICTION))]
#print(TEXTNAMES_PREDICTION)

with open(FILENAME_GROUNDTRUTH) as f:
    IMAGENAMES_GROUNDTRUTH = f.read().splitlines()
IMAGENAMES_GROUNDTRUTH = sorted(IMAGENAMES_GROUNDTRUTH,key=os.path.basename)
#print(IMAGENAME_GROUNDTRUTH)

THRESH_CONFIDENCE      = 0.1
THRESH_IOU_CONFUSION   = 0.5
THRESH_CONFIDENCE_DRAW = 0.1
SCALE_REDUCTION = 0.5
SHOW_FALSE = 0
SHOW_TOTAL = 0
SHOW_DETECTED_ONLY = 0
TIME_WAIT  = 0
SAVE_FALSE = 0
SAVE_TOTAL = 0
SAVE_DETECTED_ONLY = 0

DRAW_FALSE = SHOW_FALSE | SAVE_FALSE
DRAW_TOTAL = SHOW_TOTAL | SAVE_TOTAL
DRAW       = DRAW_FALSE | DRAW_TOTAL

NAMES_CLASS = ["plate"]
NUMBER_CLASSES = len(NAMES_CLASS)

###

label_generator = label_module.LabelGenerator(number_color = NUMBER_CLASSES+1)
image = cv2.imread(IMAGENAMES_GROUNDTRUTH[0])
height_image, width_image = image.shape[:2]
label_generator.get_legend(size=5,
                           names_class=NAMES_CLASS,
                           height_image=height_image,
                           width_image=width_image)

metric = metric_module.MAPMetric(names_class=NAMES_CLASS,
                                 check_class_first=False)

def show_image(image_labeled):
    cv2.imshow("image",image_labeled)
    key = cv2.waitKey(TIME_WAIT)
    if key == ord("q"):
       quit()

for imagename,textname_prediction in zip(IMAGENAMES_GROUNDTRUTH,TEXTNAMES_PREDICTION):
    textname_groundtruth = imagename.replace("images","labels").replace("jpg","txt")

    if DRAW:
        image = cv2.imread(imagename)
        height_image, width_image = image.shape[:2]
    else:
        width_image, height_image = imagesize.get(imagename)

    with open(textname_groundtruth) as f:
        info_groundtruth = f.read().splitlines()
    bboxes_groundtruth = []
    labels_groundtruth = []
    for bbox in info_groundtruth:
        bbox = bbox.split()
        label = int(bbox[0])
        #label = 0
        x_center    = float(bbox[1])*width_image
        y_center    = float(bbox[2])*height_image
        width_bbox  = float(bbox[3])*width_image
        height_bbox = float(bbox[4])*height_image
        bboxes_groundtruth.append([int(x_center-width_bbox/2),
                                   int(y_center-height_bbox/2),
                                   int(x_center+width_bbox/2),
                                   int(y_center+height_bbox/2)])
        labels_groundtruth.append(label)

    with open(textname_prediction) as f:
         info_prediction = f.read().splitlines()
    bboxes_prediction = []
    labels_prediction = []
    scores_prediction = []
    for bbox in info_prediction:
        bbox = bbox.split()
        label      = int(bbox[0])
        #label      = 0
        x_center    = float(bbox[1])*width_image
        y_center    = float(bbox[2])*height_image
        width_bbox  = float(bbox[3])*width_image
        height_bbox = float(bbox[4])*height_image
        confidence  = float(bbox[5])
        if confidence>=THRESH_CONFIDENCE:
            bboxes_prediction.append([int(x_center-width_bbox/2),
                                      int(y_center-height_bbox/2),
                                      int(x_center+width_bbox/2),
                                      int(y_center+height_bbox/2)])
            labels_prediction.append(label)
            scores_prediction.append(confidence)

    metric.update(bboxes_prediction=bboxes_prediction,
                  labels_prediction=labels_prediction,
                  scores_prediction=scores_prediction,
                  bboxes_groundtruth=bboxes_groundtruth,
                  labels_groundtruth=labels_groundtruth)

    if DRAW:
        print(imagename)
        #print(metric.matched)
        #print(metric.unmatched_groundtruth)
        #print(metric.unmatched_prediction)
        if DRAW_FALSE:
            image_labeled = image.copy()
            infos_groundtruth_draw = {}
            infos_prediction_draw  = {}
            for i,j in metric.matched:
                if labels_groundtruth[i] != labels_prediction[j]:
                    infos_groundtruth_draw.setdefault(tuple(bboxes_groundtruth[i]),[]).append(labels_groundtruth[i])

            for i in metric.unmatched_groundtruth:
                infos_groundtruth_draw.setdefault(tuple(bboxes_groundtruth[i]),[]).append(labels_groundtruth[i])
         
            for i,j in metric.matched:
                if labels_groundtruth[i] != labels_prediction[j] and scores_prediction[j]>THRESH_CONFIDENCE_DRAW:
                    infos_prediction_draw.setdefault(tuple(bboxes_prediction[j]),[]).append((labels_prediction[j],
                                                                                             scores_prediction[j]))
         
            for j in metric.unmatched_prediction and scores_prediction[j]>THRESH_CONFIDENCE_DRAW:
                infos_prediction_draw.setdefault(tuple(bboxes_prediction[j]),[]).append((labels_prediction[j],
                                                                                         scores_prediction[j]))

            if len(infos_prediction_draw.keys()) != 0:
                for key in infos_groundtruth_draw.keys():
                    labels = infos_groundtruth_draw.get(key,[])
                    for label in labels:
                        image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                            text="",
                                                                            size=7,
                                                                            _class=label,
                                                                            bbox_coord=key)

                for key in infos_prediction_draw.keys():
                    infos = sorted(infos_prediction_draw.get(key,[]),
                                   key=lambda x:x[1], reverse=True)
                    labels,scores = zip(*infos)
                    scores = list(scores)
                    for i,label in enumerate(reversed(labels)):
                        text = "".join([ " "+str(int(s*1e2))+" " for s in scores])
                        del scores[-1]
                        image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                            text=text,
                                                                            size=2,
                                                                            _class=label,
                                                                            bbox_coord=key)

                #image_labeled = label_generator.add_legend(image_labeled)
                height_image_legend, width_image_legend = image_labeled.shape[:2]
                height_image_show = int(height_image_legend*SCALE_REDUCTION)
                width_image_show  = int( width_image_legend*SCALE_REDUCTION)

                image_labeled = cv2.resize(image_labeled,(width_image_show,height_image_show))
                if SHOW_FALSE:
                    show_image(image_labeled)
                if SAVE_FALSE:
                    imagename_noext = os.path.splitext(os.path.basename(imagename))[0]
                    cv2.imwrite(os.path.join("results/",imagename_noext+"_false.jpg"),image_labeled)

        if DRAW_TOTAL:
            image_labeled = image.copy()
            infos_groundtruth_draw = {}
            infos_prediction_draw  = {}
            for bbox, label in zip(bboxes_groundtruth,
                                   labels_groundtruth):
                infos_groundtruth_draw.setdefault(tuple(bbox),[]).append(label)
         
            for j, bbox, label, confidence in zip(range(len(bboxes_prediction)),
                                                  bboxes_prediction,
                                                  labels_prediction,
                                                  scores_prediction):
                if scores_prediction[j]>THRESH_CONFIDENCE_DRAW:
                    infos_prediction_draw.setdefault(tuple(bboxes_prediction[j]),[]).append((labels_prediction[j],
                                                                                             scores_prediction[j]))

            if len(infos_prediction_draw.keys()) != 0:
                for key in infos_groundtruth_draw.keys():
                    labels = infos_groundtruth_draw.get(key,[])
                    for label in labels:
                        image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                            text="",
                                                                            size=7,
                                                                            _class=label,
                                                                            bbox_coord=key)

                for key in infos_prediction_draw.keys():
                    infos = sorted(infos_prediction_draw.get(key,[]),
                                   key=lambda x:x[1], reverse=True)
                    labels,scores = zip(*infos)
                    scores = list(scores)
                    for i,label in enumerate(reversed(labels)):
                        text = "".join([ " "+str(int(s*1e2))+" " for s in scores])
                        del scores[-1]
                        image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                            text=text,
                                                                            size=2,
                                                                            _class=label,
                                                                            bbox_coord=key)

                #image_labeled = label_generator.add_legend(image_labeled)
                height_image_legend, width_image_legend = image_labeled.shape[:2]
                height_image_show = int(height_image_legend*SCALE_REDUCTION)
                width_image_show  = int( width_image_legend*SCALE_REDUCTION)

                image_labeled = cv2.resize(image_labeled,(width_image_show,height_image_show))
                if SHOW_TOTAL and len(infos_prediction_draw.keys()) != 0:
                    show_image(image_labeled)
                if SAVE_TOTAL:
                    imagename_noext = os.path.splitext(os.path.basename(imagename))[0]
                    cv2.imwrite(os.path.join("results/",imagename_noext+"_total.jpg"),image_labeled)

#metric.get_mAP(type_mAP="VOC07",
#               conclude=True)
#print
metric.get_mAP(type_mAP="VOC12",
               conclude=True)
print
#metric.get_mAP(type_mAP="COCO",
#               conclude=True)
#print
metric.get_confusion(thresh_confidence=THRESH_CONFIDENCE,
                     thresh_IOU=THRESH_IOU_CONFUSION,
                     conclude=True)
