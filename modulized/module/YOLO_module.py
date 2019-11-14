from __future__ import print_function
from ctypes import *
import numpy as np
import cv2
import os
import time

dirpath = os.path.realpath(__file__)
for _ in range(3):
    dirpath = os.path.dirname(dirpath)
os.chdir(dirpath)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL(os.getcwd()+"/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

class YOLODetector():

    def __init__(self,file_cfg,file_weights,file_data,thresh_yolo,letter_box,whitelist):
        self._net = load_net(file_cfg, file_weights, 0)
        self._meta = load_meta(file_data)
        self.thresh_yolo = thresh_yolo
        self.letter_box = letter_box
        self.whitelist = whitelist
        self._get_size(file_cfg)
        self._image_darknet = make_image(self.width_yolo,self.height_yolo,3)

    def _get_size(self,file_cfg):
        with open(file_cfg) as f:
            data = f.readlines()
        self.height_yolo = -1
        self.width_yolo = -1
        for datum in data:
            datum = datum.strip("\n")
            if "#" not in datum:
                if "height" in datum:
                    self.height_yolo = int(datum.split("=")[-1])
                elif "width" in datum:
                    self.width_yolo = int(datum.split("=")[-1])
        if self.height_yolo <= 0 or self.width_yolo <= 0:
            print("fail to read width or height in .cfg file")

    def detect(self,data):
        self._preprocess(data.image_origin.copy())
        self._getBoxes(thresh=self.thresh_yolo)
        data.info["tag"] = self._results_yolo_to_bboxes()
        data.info["results"] = self._results

    def _fill_letter_box(self,image0):
        image = np.full((self.height_yolo,self.width_yolo,3),127,dtype=np.uint8)
        height_image0,width_image0,_ = image0.shape
        self._scaling = min( float( self.width_yolo)/ width_image0 , float(self.height_yolo)/height_image0 )
        height_image = int( height_image0 * self._scaling )
        width_image  = int(  width_image0 * self._scaling )
        image[:height_image,:width_image,:] = cv2.resize(image0, (width_image,height_image))
        return image

    def _preprocess(self,image):
        self.image_origin = image.copy()
        self._height_image, self._width_image = self.image_origin.shape[:2]
        if self.letter_box:
            image_resized = self._fill_letter_box(self.image_origin)
        else:
            image_resized = cv2.resize(self.image_origin,(self.width_yolo,self.height_yolo))
        self._image_bytes = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).tobytes()

    #def detect(net, meta, image_bytes, thresh=.5, hier_thresh=.5, nms=.45):
    def _getBoxes(self, thresh=.5, hier_thresh=.5, nms=.45):
        copy_image_from_bytes(self._image_darknet, self._image_bytes)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(self._net, self._image_darknet)
        dets = get_network_boxes(self._net, self._image_darknet.w, self._image_darknet.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): do_nms_obj(dets, num, self._meta.classes, nms);
    
        if self.letter_box:
            multiplier_x = multiplier_y = 1/self._scaling
        else:
            multiplier_x = float( self._width_image)/ self.width_yolo
            multiplier_y = float(self._height_image)/self.height_yolo
        results = []
        for j in range(num):
            for i in range(self._meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    results.append((self._meta.names[i], dets[j].prob[i], (b.x*multiplier_x,
                                                                           b.y*multiplier_y,
                                                                           b.w*multiplier_x,
                                                                           b.h*multiplier_y)))
        self._results = sorted(results, key=lambda x: x[1], reverse=True)
        #free_image(image_darknet)
        free_detections(dets, num)
    
    def _coord_convert(self,coord):
        x_center, y_center, x_width, y_height = coord
        x_width  = np.clip(int(x_width)              ,0 ,self._width_image )
        y_height = np.clip(int(y_height)             ,0 ,self._height_image)
        left     = np.clip(int(x_center-x_width/2.)  ,0 ,self._width_image )
        top      = np.clip(int(y_center-y_height/2.) ,0 ,self._height_image)
        return left, top, x_width, y_height
    
    def _results_yolo_to_bboxes(self):
    
        # collect detection results with the same bounding box together
        self.resultsDict = {}
        for result in self._results:
            name = result[0]
            if name in self.whitelist or len(self.whitelist)==0:
                key = result[2]
                self.resultsDict.setdefault(key,[]).append((result[0],result[1]))
    
        bboxes = []
        for key in self.resultsDict.keys():
            bbox = {}
            objectPicX, objectPicY, objectWidth, objectHeight = self._coord_convert(key) # key is coordinate of bounding box
            if objectWidth==0 or objectHeight==0:
                break
            bbox["objectPicX"] = objectPicX
            bbox["objectPicY"] = objectPicY
            bbox["objectWidth"] = objectWidth
            bbox["objectHeight"] = objectHeight
            objectTypes, confidences = zip(*self.resultsDict[key])
            bbox["objectTypes"] = list(objectTypes)
            bbox["confidences"] = [float("%.3f"%(confidence*100.)) for confidence in confidences]
            bboxes.append(bbox)
        return bboxes

    def _draw_label(self, data, text, left, top, right, bottom):
        cv2.rectangle(data.image_labeled,(left,top),(right,bottom),(  0,  0,  0,),3)
        cv2.rectangle(data.image_labeled,(left,top),(right,bottom),(  0,  0,255,),2)
        text = str(text).upper()
        cv2.putText(data.image_labeled, text, (left+15,top-15), cv2.FONT_HERSHEY_COMPLEX, 1, (  0,   0,   0), 2)
        cv2.putText(data.image_labeled, text, (left+15,top-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    def _check_draw_labels(self,data):
        if not data.drawed:
            #for bbox in data.json_label.get("tag",[]):
            for bbox in data.info.get("tag",[]):
                left   = bbox.get("objectPicX")
                top    = bbox.get("objectPicY")
                right  = bbox.get("objectWidth")+left
                bottom = bbox.get("objectHeight")+top
                text   = " ".join(bbox.get("objectTypes"))
                self._draw_label(data, text, left, top, right, bottom)
            data.drawed = True
    
    def show(self,data,draw_labels=None,to_draw=False,time_wait=1):
        if to_draw and not data.drawed:
            if draw_labels:
                draw_labels(data)
            else:
                self._check_draw_labels(data)
        cv2.imshow("image",
                   data.image_labeled if to_draw else data.image_origin)
        key = cv2.waitKey(time_wait) & 0xff
        if key == ord("q"):
            quit()
        if time_wait == 0:
            cv2.destroyWindow("image")
 
    def save(self,data,to_draw):
        if to_draw and not self.drawed:
            self._check_draw_labels()
        cv2.imwrite(self.filename,
                    data.image_labeled if to_draw else data.image_origin)

if __name__ == "__main__":

    import data_manager_module

    # settings for YOLO model

    # yolov3
    #file_cfg = b"cfg/yolov3.cfg"
    #file_weights = b"weights/yolov3.weights"
    #file_data = b"cfg/coco.data"

    # yolov3-spp
    #file_cfg = b"cfg/yolov3-spp.cfg"
    #file_weights = b"weights/yolov3-spp.weights"
    #file_data = b"cfg/coco.data"

    # yolov3-tiny
    file_cfg = b"cfg/yolov3-tiny.cfg"
    file_weights = b"../weights/yolov3-tiny.weights"
    file_data = b"cfg/coco.data"

    # yolov3-tiny for license plate
    #file_cfg = b"cfg/yolov3_tiny_car_plate02.cfg"
    #file_weights = b"weights/yolov3_tiny_car_plate03_final.weights"
    #file_data = b"cfg/car_plate.data"

    # settings
    thresh_yolo = 0.1
    #letter_box = False
    letter_box = True

    # prediction with some constraint
    # if whitelist is empty, detect everything
    #whitelist = ["car","motorbike","bus", "truck"]
    whitelist = []

    image = cv2.imread("data/dog.jpg")

    yolo = YOLODetector(file_cfg,
                        file_weights,
                        file_data,
                        thresh_yolo,
                        letter_box,
                        whitelist)
    # prediction
    for _ in range(100):

        # prediction starts
        time0 = time.time()

        data = data_manager_module.DataManager(image=image)

        # YOLO
        yolo.detect(data)
        #yolo.save(to_draw=False)
        yolo.show(data,to_draw=True)
        print(data.info)

        # prediction ends
        print("duration %.6f s"%(time.time()-time0))

    del yolo
