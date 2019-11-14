from __future__ import print_function
import numpy as np
import cv2
import os
from copy import deepcopy
from module import labels_module, metric_module, printer_with_saving_module


# settings

#DIRNAME_TEST = "compare_recycle/darknet_train_pt_recycle00_run00_final_h352_w608/test000"
DIRNAME_TEST = "compare_recycle/darknet_train_recycle09_run00_it30200_h352_w608/test000"
#DIRNAME_TEST = "compare_recycle/darknet_train_recycle06_run00_it30200_h352_w608/pseudolabel000/"
#DIRNAME_TEST = "compare_recycle/darknet_train_yolov3_tiny_recycle03_run01_it15200_h352_w608/test000"
DIRNAME_PREDICTION = os.path.join(DIRNAME_TEST,"labels_prediction")
FILENAME_GROUNDTRUTH = "test000.txt"
#FILENAME_GROUNDTRUTH = "test000_addfnv01.txt"
#FILENAME_GROUNDTRUTH = "pseudo001.txt"
FILENAME_RECORD = os.path.join( DIRNAME_TEST , os.path.splitext(FILENAME_GROUNDTRUTH)[0]+"_record.txt" )
FILENAME_GROUNDTRUTH = os.path.join(DIRNAME_TEST,FILENAME_GROUNDTRUTH)
prints = printer_with_saving_module.PrinterWithSaving(filename=FILENAME_RECORD).prints
#prints(DIRNAME_TEST)

DICT_TEXTNAMES_PREDICTION = { os.path.splitext(p)[0] : os.path.join(DIRNAME_PREDICTION,p) for p in os.listdir(DIRNAME_PREDICTION)}
#print(DICT_TEXTNAMES_PREDICTION)

with open(FILENAME_GROUNDTRUTH) as f:
    IMAGENAMES_GROUNDTRUTH = f.read().splitlines()
#print(IMAGENAMES_GROUNDTRUTH)

THRESH_CONFIDENCE      = 0.1
THRESH_IOU_CONFUSION   = 0.5
THRESH_CONFIDENCE_DRAW = 0.1
SCALE_REDUCTION = 0.5
SHOW_FALSE = 0
SHOW_TOTAL = 0
TIME_WAIT  = 0
SAVE_FALSE = 0
SAVE_TOTAL = 0

DRAW_FALSE = ( SHOW_FALSE | SAVE_FALSE )
DRAW_TOTAL = ( SHOW_TOTAL | SAVE_TOTAL )
DRAW       = ( DRAW_FALSE | DRAW_TOTAL )

NAMES_CLASS = ["PET",
               "oil",
               "soy",
               "colored",
               "other"]
#NAMES_CLASS = ["Obj"]
NUMBER_CLASSES = len(NAMES_CLASS)
#STRATEGY = "MAX_SCORE"
STRATEGY = "NON_PET_FIRST"
#STRATEGY = "NONE"

###
prints("# of data: %d"%len(IMAGENAMES_GROUNDTRUTH))

label_generator = labels_module.LabelGenerator(number_color = NUMBER_CLASSES+1)
image = cv2.imread(IMAGENAMES_GROUNDTRUTH[0])
height_image, width_image = image.shape[:2]
label_generator.get_legend(size=5,
                           names_class=NAMES_CLASS,
                           height_image=height_image,
                           width_image=width_image)

metric = metric_module.ObjectDetectionMetric(names_class=NAMES_CLASS,
                                             check_class_first=False)

def convert_yolo_train_to_opencv(coord_yolo_train,height_image,width_image):
    x_center, y_center, w_bbox, h_bbox = coord_yolo_train
    left   = (x_center-w_bbox/2.)*width_image
    top    = (y_center-h_bbox/2.)*height_image
    right  = (x_center+w_bbox/2.)*width_image
    bottom = (y_center+h_bbox/2.)*height_image
    coord_opencv = left, top, right, bottom
    return [int(c) for c in coord_opencv]

def show_image(image_labeled):
    #cv2.namedWindow("image",0)
    cv2.imshow("image",image_labeled)
    key = cv2.waitKey(TIME_WAIT)
    if key == ord("q"):
       quit()

def decide_prediction(bboxes_prediction,
                      labels_prediction,
                      scores_prediction,
                      STRATEGY):

    if STRATEGY in ["MAX_SCORE","NON_PET_FIRST"]:
        infos_prediction = {}
        for j in range(len(labels_prediction)):
            infos_prediction.setdefault( tuple(bboxes_prediction[j]) , [] ).append(\
                [j,labels_prediction[j],scores_prediction[j]] )
        bboxes_prediction = []
        labels_prediction = []
        scores_prediction = []
        for bbox,info in infos_prediction.items():
            info = sorted(info, key=lambda x:x[2])
            bboxes_prediction.append(list(bbox))
            order = -1
            if STRATEGY == "NON_PET_FIRST" and len(info)>1 and info[-1][1]==0:
                order = -2
            labels_prediction.append(info[order][1])
            scores_prediction.append(info[order][2])
    return bboxes_prediction, labels_prediction, scores_prediction

def get_pollution_rate(content):
    data = content.split("\n")
    record = []
    for datum in data:
        datum = datum.split("]")
        if len(datum) == 2:
            datum = datum[-1].split()[0]
            record.append(int(datum))
        if len(record) == 5:
            break
    return float(sum(record[1:])) / sum(record)

for index in range(len(IMAGENAMES_GROUNDTRUTH)):
    imagename = IMAGENAMES_GROUNDTRUTH[index]
    textname_prediction = DICT_TEXTNAMES_PREDICTION[ os.path.splitext( os.path.basename(imagename) )[0] ]
    textname_groundtruth = imagename.replace("images","labels").replace("jpg","txt")

    if DRAW:
        image = cv2.imread(imagename)
        height_image, width_image = image.shape[:2]

    with open(textname_groundtruth) as f:
        info_groundtruth = f.read().splitlines()
    bboxes_groundtruth = []
    labels_groundtruth = []
    for bbox in info_groundtruth:
        bbox = bbox.split()
        label = int(bbox[0])
        #label = 0
        bboxes_groundtruth.append([float(c) for c in bbox[1:5]])
        labels_groundtruth.append(label)

    with open(textname_prediction) as f:
         info_prediction = f.read().splitlines()
    bboxes_prediction = []
    labels_prediction = []
    scores_prediction = []
    for bbox in info_prediction:
        bbox = bbox.split()
        label = int(bbox[0])
        #label = 0
        confidence = float(bbox[5])
        if confidence>=THRESH_CONFIDENCE:
            bboxes_prediction.append([float(c) for c in bbox[1:5]])
            labels_prediction.append(label)
            scores_prediction.append(confidence)
    bboxes_prediction, labels_prediction, scores_prediction = \
    decide_prediction(bboxes_prediction=bboxes_prediction,
                      labels_prediction=labels_prediction,
                      scores_prediction=scores_prediction,
                      STRATEGY=STRATEGY)

    metric.update(bboxes_prediction=bboxes_prediction,
                  labels_prediction=labels_prediction,
                  scores_prediction=scores_prediction,
                  bboxes_groundtruth=bboxes_groundtruth,
                  labels_groundtruth=labels_groundtruth)

    if not DRAW:
        continue
    print(imagename)
    #print("\r"+imagename,end="")
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

        for j in metric.unmatched_prediction:
            if scores_prediction[j]>THRESH_CONFIDENCE_DRAW:
                infos_prediction_draw.setdefault(tuple(bboxes_prediction[j]),[]).append((labels_prediction[j],
                                                                                         scores_prediction[j]))

        for key in infos_groundtruth_draw.keys():
            coord_opencv = convert_yolo_train_to_opencv(key,height_image,width_image)
            labels = infos_groundtruth_draw.get(key,[])
            for label in labels:
                image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                    text="",
                                                                    size=7,
                                                                    _class=label,
                                                                    bbox_coord=coord_opencv)

        for key in infos_prediction_draw.keys():
            coord_opencv = convert_yolo_train_to_opencv(key,height_image,width_image)
            infos = sorted(infos_prediction_draw.get(key,[]),
                           key=lambda x:x[1],reverse=True)
            labels,scores = zip(*infos)
            scores = list(scores)
            for i,label in enumerate(reversed(labels)):
                text = "".join([ " "+str(int(s*1e2))+" " for s in scores])
                del scores[-1]
                image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                    text=text,
                                                                    size=2,
                                                                    _class=label,
                                                                    bbox_coord=coord_opencv)

        image_labeled = label_generator.add_legend(image_labeled)
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

        for key in infos_groundtruth_draw.keys():
            coord_opencv = convert_yolo_train_to_opencv(key,height_image,width_image)
            labels = infos_groundtruth_draw.get(key,[])
            for label in labels:
                image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                    text="",
                                                                    size=7,
                                                                    _class=label,
                                                                    bbox_coord=coord_opencv)

        for key in infos_prediction_draw.keys():
            coord_opencv = convert_yolo_train_to_opencv(key,height_image,width_image)
            infos = sorted(infos_prediction_draw.get(key,[]),
                           key=lambda x:x[1],reverse=True)
            labels,scores = zip(*infos)
            scores = list(scores)
            for i,label in enumerate(reversed(labels)):
                text = "".join([ " "+str(int(s*1e2))+" " for s in scores])
                del scores[-1]
                image_labeled = label_generator.draw_bbox_with_text(image_labeled,
                                                                    text=text,
                                                                    size=2,
                                                                    _class=label,
                                                                    bbox_coord=coord_opencv)

        image_labeled = label_generator.add_legend(image_labeled)
        height_image_legend, width_image_legend = image_labeled.shape[:2]
        height_image_show = int(height_image_legend*SCALE_REDUCTION)
        width_image_show  = int( width_image_legend*SCALE_REDUCTION)

        image_labeled = cv2.resize(image_labeled,(width_image_show,height_image_show))
        detected = len(infos_prediction_draw.keys()) != 0
        if SHOW_TOTAL==1 or ( SHOW_TOTAL==-1 and detected ):
            show_image(image_labeled)
        if SAVE_TOTAL==1 or ( SAVE_TOTAL==-1 and detected ):
            imagename_noext = os.path.splitext(os.path.basename(imagename))[0]
            cv2.imwrite(os.path.join("results/",imagename_noext+"_total.jpg"),image_labeled)
prints("")

#content = metric.get_mAP(type_mAP="VOC07",
#                         conclude=False)
#prints(content+"\n")
content = metric.get_mAP(type_mAP="VOC12",
                         conclude=False)
prints(content+"\n")
#content = metric.get_mAP(type_mAP="COCO",
#                         conclude=False)
#prints(content+"\n")
content = metric.get_confusion(thresh_confidence=THRESH_CONFIDENCE,
                               thresh_IOU=THRESH_IOU_CONFUSION,
                               conclude=False)
prints(content+"\n")

rate_pollution = get_pollution_rate(content)
prints("pollution rate:  %.2f %%"%(1e2*rate_pollution))
