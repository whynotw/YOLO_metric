import numpy as np
import cv2
import os
from progressbar import ProgressBar
from module import labels_module, metric_module

# settings

DIRNAME_TEST = "compare_coco80/darknet_train_yolov3_spp_608/test000"
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

#NAMES_CLASS = [str(n) for n in range(80)]
with open("data/coco.names") as f:
    NAMES_CLASS = f.read().splitlines()
NUMBER_CLASSES = len(NAMES_CLASS)

###

label_generator = labels_module.LabelGenerator(number_color = NUMBER_CLASSES+1)
image = cv2.imread(IMAGENAMES_GROUNDTRUTH[0])
height_image, width_image = image.shape[:2]
label_generator.get_legend(size=3,
                           names_class=NAMES_CLASS,
                           height_image=height_image,
                           width_image=width_image)

metric = metric_module.ObjectDetectionMetric(names_class=NAMES_CLASS,
                                             check_class_first=False)

pbar = ProgressBar().start()
for index in range(len(TEXTNAMES_PREDICTION)):
    imagename = IMAGENAMES_GROUNDTRUTH[index]
    textname_prediction = TEXTNAMES_PREDICTION[index]
    textname_groundtruth = imagename.replace("images","labels").replace("jpg","txt")

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
        label      = int(bbox[0])
        #label      = 0
        confidence = float(bbox[5])
        if confidence>=THRESH_CONFIDENCE:
            bboxes_prediction.append([float(c) for c in bbox[1:5]])
            labels_prediction.append(label)
            scores_prediction.append(confidence)

    metric.update(bboxes_prediction=bboxes_prediction,
                  labels_prediction=labels_prediction,
                  scores_prediction=scores_prediction,
                  bboxes_groundtruth=bboxes_groundtruth,
                  labels_groundtruth=labels_groundtruth)
    progress = 100*index/len(TEXTNAMES_PREDICTION)
    pbar.update(progress)
pbar.finish()

#metric.get_mAP(type_mAP="VOC07",
#               conclude=True)
#print
metric.get_mAP(type_mAP="VOC12",
               conclude=True)
print
metric.get_mAP(type_mAP="COCO",
               conclude=True)
#print
#metric.get_confusion(thresh_confidence=THRESH_CONFIDENCE,
#                     thresh_IOU=THRESH_IOU_CONFUSION,
#                     conclude=True)
