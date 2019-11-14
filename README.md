# Object Detection Metric for YOLO

Calculate mean Average Precision (mAP) and confusion matrix for object detection models. Bounding box information for groundtruth and prediction is YOLO training dataset format.

Based on https://github.com/AlexeyAB/darknet to predict testing dataset.

# Prerequisites

```
python2.7
numpy
cv2
progressbar
sklearn
```

# Compile

Just `make clean` and `make` to compile darknet.

# Dataset settings

Directory to save results: `DIRNAME_TEST`

Put testing data list to `test.txt` in `DIRNAME_TEST`. Each line in `test.txt` is a path of an `.jpg` image. 
Also put its `.txt` file of label information to the associated path of in YOLO-style 
(replace directory name `images` to `labels` and replace file extension `.jpg` to `.txt`).

NOTICE: Testing data CANNOT share the same filename.

# Inference

Change settings in `modulized/save_label_as_yolo_format.py`, including:

Model settings: `FILE_CFG`, `FILE_WEIGHTS`, `FILE_DATA` and `THRESH_YOLO`
(Small `THRESH_YOLO` is suggested, because we can change `THRESH_CONFIDENCE` from small to large to get different evaluation results)

Directory to save results: `DIRNAME_TEST`

---

Use `python modulized/save_label_as_yolo_format.py` to get the inference result, which will be saved in `{DIRNAME_TEST}/labels_prediction`.


# Calculate mean Average Precision (mAP) and confusion matrix

Change settings in `modulized/compare_simple.py`, including:

Directory containing saved results: `DIRNAME_TEST`

Threshold of confidence for calculating mAP and confusion matrix: `THRESH_CONFIDENCE`

Threshold of IOU for calculating confusion matrix: `THRESH_IOU_CONFUSION`

---

Comment and uncomment last part of `modulized/compare_simple.py` to control desired the results (`metric.get_mAP` and `metric.get_confusion`).

Use `python modulized/compare_simple.py` to show the results like:

## mAP
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Mean Average Precision
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
metric: VOC12
[        person]      AP:  73.04 %   #:  89202 ( 30.37 %)
[       bicycle]      AP:  55.88 %   #:   2474 (  0.84 %)
[           car]      AP:  66.31 %   #:  15084 (  5.14 %)
[     motorbike]      AP:  66.19 %   #:   3088 (  1.05 %)
[     aeroplane]      AP:  80.10 %   #:   1447 (  0.49 %)
[           bus]      AP:  83.28 %   #:   2029 (  0.69 %)
...
[         total]     mAP:  62.27 %   #: 293740 (100.00 %)
[      weighted]     mAP:  63.04 %
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

`[total]` is directly average the AP of all categories and `[weighted]` is using number of objects of each categories to weighted average the AP.

`#` means object number of each category and its percentage.

Three types of mAP can be evaluated, including `VOC07`, `VOC12` and `COCO`.

---

## Confusion matrix
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Confusion Matrix
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
thresh_confidence: 0.1
                                                     Prediction
                      [     person] [  bicycle] [      car] [motorbike] [aeroplane] ... [   none] [  total] 
            [   person]        1438           3           1           9           4 ...        98      1553 
            [  bicycle]          17          50           0           2           5 ...         2        76 
            [      car]           8           2          14           0           0 ...         2        26 
Groundtruth [motorbike]          22           0           0         196           0 ...        28       246 
            [aeroplane]          16           2           0           2          89 ...        33       142 
...
            [     none]         246           9           5          23          33 
            [    total]        1747          66          20         232         131 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
[        person]   precision:  57.77 %     recall:  80.46 %     avg IOU:  74.20 %
[       bicycle]   precision:  56.30 %     recall:  64.27 %     avg IOU:  67.64 %
[           car]   precision:  57.17 %     recall:  75.48 %     avg IOU:  72.23 %
[     motorbike]   precision:  70.02 %     recall:  70.63 %     avg IOU:  72.51 %
[     aeroplane]   precision:  83.95 %     recall:  81.69 %     avg IOU:  78.48 %
[           bus]   precision:  75.50 %     recall:  85.36 %     avg IOU:  83.96 %
...
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

Where `none` in `Groundtruth` means false positive and `none` in `Prediction` means false negative.

# Object Detection Metric for YOLO

Calculate mean Average Precision (mAP) and confusion matrix for object detection models. Bounding box information for groundtruth and prediction is YOLO training dataset format.

Based on https://github.com/AlexeyAB/darknet to predict testing dataset.

# Prerequisites

```
python2.7
numpy
cv2
progressbar
sklearn
```

# Compile

Just `make clean` and `make` to compile darknet.

# Dataset settings

Directory to save results: `DIRNAME_TEST`

Put testing data list to `test.txt` in the directory `DIRNAME_TEST`. Each line in `test.txt` is a path of a `.jpg` image. 
Also put its `.txt` file (label information) to the associated path of in YOLO-style 
(replace directory name `images` to `labels` and replace file extension `.jpg` to `.txt`).

# Inference

Change settings in `modulized/save_label_as_yolo_format.py`, including:

Model settings: `FILE_CFG`, `FILE_WEIGHTS`, `FILE_DATA` and `THRESH_YOLO`
(Small `THRESH_YOLO` is suggested, because we can tune `THRESH_CONFIDENCE` from small to large values during evalution to get different results)

Directory to save results: `DIRNAME_TEST`

---

Use `python modulized/save_label_as_yolo_format.py` to get the inference result, which will be saved in directory `{DIRNAME_TEST}/labels_prediction`.

# Calculate mean Average Precision (mAP) and confusion matrix

Change settings in `modulized/compare_simple.py`, including:

Directory containing saved results: `DIRNAME_TEST`

Threshold of confidence for calculating mAP and confusion matrix: `THRESH_CONFIDENCE`

Threshold of IOU for calculating confusion matrix: `THRESH_IOU_CONFUSION`

---

Comment or uncomment the last part of `modulized/compare_simple.py` to control desired the results (`metric.get_mAP` and `metric.get_confusion`).

Use `python modulized/compare_simple.py` to show the results like:

## mAP
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Mean Average Precision
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
metric: VOC12
[        person]      AP:  73.04 %   #:  89202 ( 30.37 %)
[       bicycle]      AP:  55.88 %   #:   2474 (  0.84 %)
[           car]      AP:  66.31 %   #:  15084 (  5.14 %)
[     motorbike]      AP:  66.19 %   #:   3088 (  1.05 %)
[     aeroplane]      AP:  80.10 %   #:   1447 (  0.49 %)
[           bus]      AP:  83.28 %   #:   2029 (  0.69 %)
...
[         total]     mAP:  62.27 %   #: 293740 (100.00 %)
[      weighted]     mAP:  63.04 %
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

`[total] mAP` is directly average the AP of all categories and `[weighted] mAP` is using number of objects of each category to weighted average the AP.

`#` means object number of each category and its percentage.

Three types of mAP can be evaluated, including `VOC07`, `VOC12` and `COCO`.

---

## Confusion matrix
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Confusion Matrix
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
thresh_confidence: 0.1
                                                     Prediction
                      [     person] [  bicycle] [      car] [motorbike] [aeroplane] ... [   none] [  total] 
            [   person]        1438           3           1           9           4 ...        98      1553 
            [  bicycle]          17          50           0           2           5 ...         2        76 
            [      car]           8           2          14           0           0 ...         2        26 
Groundtruth [motorbike]          22           0           0         196           0 ...        28       246 
            [aeroplane]          16           2           0           2          89 ...        33       142 
...
            [     none]         246           9           5          23          33 
            [    total]        1747          66          20         232         131 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
[        person]   precision:  57.77 %     recall:  80.46 %     avg IOU:  74.20 %
[       bicycle]   precision:  56.30 %     recall:  64.27 %     avg IOU:  67.64 %
[           car]   precision:  57.17 %     recall:  75.48 %     avg IOU:  72.23 %
[     motorbike]   precision:  70.02 %     recall:  70.63 %     avg IOU:  72.51 %
[     aeroplane]   precision:  83.95 %     recall:  81.69 %     avg IOU:  78.48 %
[           bus]   precision:  75.50 %     recall:  85.36 %     avg IOU:  83.96 %
...
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```
Where the `none` row of `Groundtruth` means false positive and the `none` cloumn of `Prediction` means false negative.
