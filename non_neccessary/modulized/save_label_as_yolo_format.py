import cv2
import errno
import os
import time
from module import YOLO_module, data_manager_module


# settings for YOLO model

# yolov3
#FILE_CFG = b"cfg/yolov3.cfg"
#FILE_WEIGHTS = b"weights/yolov3.weights"
#FILE_DATA = b"cfg/coco.data"

# yolov3-spp
#FILE_CFG = b"cfg/yolov3-spp.cfg"
#FILE_WEIGHTS = b"../weights/yolov3-spp.weights"
#FILE_DATA = b"cfg/coco.data"

# yolov3-tiny
#FILE_CFG = b"cfg/yolov3-tiny.cfg"
#FILE_WEIGHTS = b"../weights/yolov3-tiny.weights"
#FILE_DATA = b"cfg/coco.data"

# yolov3-tiny for license plate
#FILE_CFG = b"cfg/yolov3_tiny_car_plate02.cfg"
#FILE_WEIGHTS = b"weights/yolov3_tiny_car_plate03_final.weights"
#FILE_DATA = b"cfg/car_plate.data"

# yolov3-spp for recycle
FILE_CFG = b"cfg/yolov3-spp_recycle00.cfg"
FILE_WEIGHTS = b"../darknet_train_recycle05_run00/backup/yolov3-spp_recycle00_final.weights"
FILE_DATA = b"../darknet_train_recycle00_run00/cfg/recycle.data"

# yolov3-tiny for generated license plate
#FILE_CFG = b"cfg/yolov3-tiny_plate00.cfg"
#FILE_WEIGHTS = b"../darknet_train_yolov3_tiny_car_plate_generated09/backup/yolov3-tiny_plate00_4000.weights"
#FILE_DATA = b"cfg/plate.data"

# settings
THRESH_YOLO = 0.1

# prediction with some constraint
# if whitelist is empty, detect everything
#WHITELIST = ["car","motorbike","bus", "truck"]
WHITELIST = []

DIRNAME_TEST = "compare_recycle/darknet_train_recycle05_run00_15200/test000"
#DIRNAME_TEST = "compare_plate/darknet_train_yolov3_tiny_car_plate_generated09_4000/test001"
#DIRNAME_TEST = "compare_coco80/darknet_train_yolov3_tiny_416/test000"
#DIRNAME_TEST = "compare_coco80/darknet_train_yolov3_spp_608/test000"

dirnames_mkdir = []
dirname_test_parent = DIRNAME_TEST
while len(dirname_test_parent) != 0:
    if not os.path.isdir(dirname_test_parent):
        dirnames_mkdir.insert(0,dirname_test_parent)
    dirname_test_parent = os.path.dirname(dirname_test_parent)
for dirname in dirnames_mkdir:
    os.makedirs(dirname)

TEXTNAME_TEST = os.path.join(DIRNAME_TEST, "test.txt")
with open(TEXTNAME_TEST) as f:
    IMAGENAMES = f.read().splitlines()

# get class names
with open(FILE_DATA) as f:
    data = f.read().splitlines()
for datum in data:
    if len(datum) == 0:
        continue
    if "#" == datum[0]:
        continue
    datum = datum.split("=")
    if datum[0].strip(" ") == "names":
        FILE_NAMES = datum[1].strip(" ")
        break
with open(FILE_NAMES) as f:
    NAMES_CLASS = f.read().splitlines()
DICT_CLASS = { NAMES_CLASS[i]:i for i in range(len(NAMES_CLASS))}

###

yolo = YOLO_module.YOLODetector(FILE_CFG,
                                FILE_WEIGHTS,
                                FILE_DATA,
                                THRESH_YOLO,
                                WHITELIST)

def yolo_output_to_yolo_train(results_yolo_output,height_yolo,width_yolo):
    results_train = []
    for result in results_yolo_output:
        coord_yolo_output = result[2]
        coord_yolo_train = [coord_yolo_output[0]/ width_yolo,
                            coord_yolo_output[1]/height_yolo,
                            coord_yolo_output[2]/ width_yolo,
                            coord_yolo_output[3]/height_yolo]

        coord_yolo_train = ["%.05f"%x for x in coord_yolo_train]

        label = DICT_CLASS[result[0]]
        results_train.append([label]+coord_yolo_train+["%.08f"%result[1]])
    return results_train

DIRNAME_SAVE = os.path.join(DIRNAME_TEST,"labels_prediction")
try:
    os.makedirs(DIRNAME_SAVE)
except:
    pass

for imagename in IMAGENAMES:
    time0 = time.time()

    print(imagename)
    image = cv2.imread(imagename)
    data = data_manager_module.DataManager(image=image)                                                                                                                                                     

    # YOLO
    yolo.detect(data)
    #print(data.info)
    results_yolo_output = yolo_output_to_yolo_train(data.info["results"],yolo._height_yolo,yolo._width_yolo)
    #print(results_output)
    with open(os.path.join( DIRNAME_SAVE, os.path.basename(imagename).replace("jpg","txt") ),"w") as f:
        for result in results_yolo_output:
            #print(" ".join([str(s) for s in result]))
            f.write(" ".join([str(s) for s in result])+"\n")

    #yolo.save(to_draw=False)
    #yolo.save(to_draw=True)

    #yolo._checkDrawLabels(data)
    #yolo.show(data=data,to_draw=True,time_wait=0)
    #cv2.imshow("image",cv2.resize(data.image_labeled,(500,500)))
    #cv2.waitKey(0)

    # prediction ends
    print("duration %.6f s"%(time.time()-time0))
del yolo
