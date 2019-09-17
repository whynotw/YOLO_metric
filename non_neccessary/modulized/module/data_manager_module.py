import datetime
import cv2

class DataManager():

    def __init__(self,
                 image,
                 info={},
                 json_label={}):

        self.image_origin = image.copy()
        self.image_labeled = image.copy()
        self.drawed = False
        self.datetime_created = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
        self.info = info
        self.json_label = json_label

    @property
    def time_process(self):
        try:
            self._time_process
        except:
            self._time_process = \
            (   datetime.datetime.now()
              - datetime.datetime.strptime(self.datetime_created,"%Y%m%d-%H%M%S.%f")
            ).total_seconds()
        finally:
            return self._time_process

    def _draw(self,draw_labels):
        if not self.drawed:
            if draw_labels:
                draw_labels(self.image_labeled,self.json_label)
            self.drawed = True

    def set_image(self,image):
        self.image_origin = image.copy()
        self.image_labeled = image.copy()

    def reset_image_labeled(self):
        self.drawed = False
        self.image_labeled = self.image_origin.copy()

    def show(self,
             draw_labels=None,
             time_wait=1,
             to_draw=True):

        if to_draw:
            self._draw(draw_labels)
        cv2.imshow("image",
                   self.image_labeled if to_draw else self.image_origin)
        key = cv2.waitKey(time_wait) & 0xff
        if key == ord("q"):
            quit()
        if time_wait == 0:
            cv2.destroyWindow("image")
 
    def save(self,
             filename,
             draw_labels=None,
             to_draw=True):

        if to_draw:
            self._draw(draw_labels)
        cv2.imwrite(filename,
                    self.image_labeled if to_draw else self.image_origin)

if __name__ == "__main__":

    import numpy as np
    import time

    def draw_labels(image_labeled,json_label):
        height, width, _ = image_labeled.shape
        range_cover = 0.8
        x0 = int(height*(1.-range_cover)/2.)
        y0 = int(width *(1.-range_cover)/2.)
        x1 = width-x0
        y1 = height-y0
        image_labeled[y0:y1,x0:x1,:] = np.uint(range_cover*255)

    size = 100
    for _ in range(10):
        image = np.zeros((size,size,3),np.uint8)

        #time0 = time.time()
        data = DataManager(image=image)
        #print(time.time()-time0)

        if len(data.json_label.get("tag",[])):
            data.save(draw_labels,"data.jpg")

        data.show(draw_labels=draw_labels)#,time_wait=0)

        print(data.json_label)
        print(data.time_process)
        del data
