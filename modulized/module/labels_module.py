import numpy as np
import cv2
import os
import string

class LabelGenerator():

    def __init__(self,number_color):

        self.letters = string.printable[:-5]
        self.number_color = number_color
        self._get_pixel_value()
        self._generate_templates()

    def draw_bbox_with_text(self,
                            image_origin,
                            text,
                            size,
                            _class,
                            bbox_coord):

        if _class >= self.number_color:
            print("class number %d out of total classes 0~%d"%(_class,self.number_color))
            _class %= self.number_color

        size_line = size
        image_labeled = image_origin.copy()
        height_image, width_image = image_origin.shape[:2]

        left_bbox   = np.clip(bbox_coord[0]          ,0, width_image-size_line)
        top_bbox    = np.clip(bbox_coord[1]          ,0,height_image-size_line)
        right_bbox  = np.clip(bbox_coord[2]-size_line,0, width_image-size_line)
        bottom_bbox = np.clip(bbox_coord[3]-size_line,0,height_image-size_line)
        subimage_text = self.make_text_label(text,size,_class)
        height_subimage, width_subimage = subimage_text.shape[:2]
        if height_subimage > top_bbox:
            top_subimage = np.clip(top_bbox,0,height_image-height_subimage)
        else:
            top_subimage = np.clip(top_bbox-height_subimage+size_line,0,height_image-height_subimage)
        #top_subimage = np.clip(top_bbox-height_subimage+size_line,0,height_image-height_subimage)
        left_subimage   = left_bbox
        right_subimage  = np.clip(left_subimage+ width_subimage,0, width_image)
        bottom_subimage = np.clip( top_subimage+height_subimage,0,height_image)

        #cv2.rectangle(image_labeled,(left,top),(right,bottom),self.pixels_bg[_class],size*2)
        image_labeled[   top_bbox:   top_bbox+size_line, left_bbox:right_bbox+size_line,:] = self.pixels_bg[_class]
        image_labeled[bottom_bbox:bottom_bbox+size_line, left_bbox:right_bbox+size_line,:] = self.pixels_bg[_class]
        image_labeled[   top_bbox:bottom_bbox+size_line, left_bbox: left_bbox+size_line,:] = self.pixels_bg[_class]
        image_labeled[   top_bbox:bottom_bbox+size_line,right_bbox:right_bbox+size_line,:] = self.pixels_bg[_class]
        image_labeled[ top_subimage:bottom_subimage,left_subimage: right_subimage,:] = \
        subimage_text[:bottom_subimage-top_subimage,:right_subimage-left_subimage,:]
        return image_labeled

    def add_legend(self,image_labeled):
        height_image, width_image = image_labeled.shape[:2]
        image_labeled0 = image_labeled.copy()
        height_image_legend = self.image_legend.shape[0]
        image_labeled_legend = np.empty((height_image+height_image_legend,width_image,3),dtype=np.uint8)
        image_labeled_legend[:height_image,:,:] = image_labeled
        image_labeled_legend[height_image:,:,:] = self.image_legend
        return image_labeled_legend

    def make_text_label(self,
                        characters,
                        size,
                        _class):

        if _class >= self.number_color:
            print("class number %d out of total classes 0~%d"%(_class,self.number_color))
            _class %= self.number_color
        if size > 7:
            print("size should not larger than 7")
            size = 7

        if not characters:
            return np.empty((0,0,3),np.uint8)
        subimages = [self.templates[size][c if c in self.letters else " "] for c in characters]
        image_grayscale = np.hstack(subimages)

        if self.bright_bg[_class]:
            image_bg =       np.expand_dims(image_grayscale,axis=-1)/51 * (   np.uint8(self.pixels_bg[_class].reshape(1,3))/5)
        else:
            image_bg = 255 - np.expand_dims(image_grayscale,axis=-1)/51 * (51-np.uint8(self.pixels_bg[_class].reshape(1,3))/5)
        return image_bg

    def _get_pixel_value(self):
        coefficient_grayscale = np.array([0.114,0.587,0.299])
        pixel_hsv = np.empty((1,1,3),dtype=np.uint8)
        pixel_hsv[0,0,1] = 255
        pixel_hsv[0,0,2] = 255
        self.pixels_bg = np.empty((self.number_color,3),dtype=int)
        for i in range(self.number_color):
            pixel_hsv[0,0,0] = i*180/self.number_color
            pixel_bgr = cv2.cvtColor(pixel_hsv,cv2.COLOR_HSV2BGR)
            self.pixels_bg[i,:] = (pixel_bgr/5)*5
        self.bright_bg = np.sum(self.pixels_bg*coefficient_grayscale, axis=-1) > 127

    def get_legend(self,
                   size,
                   names_class,
                   height_image,
                   width_image):
        height_subimage = self.templates[size]["0"].shape[0]
        size_line = size
        image_legend = np.ones(((height_subimage+size_line)*len(names_class)+size_line,width_image,3),dtype=np.uint8)*128
        x_coord = 0
        y_coord = size_line
        for i,label in enumerate(names_class):
            subimage = self.make_text_label(characters=label,
                                            size=size,
                                            _class=i)
            height_subimage, width_subimage = subimage.shape[:2]
            if x_coord+width_subimage>=width_image and x_coord!=0:
                x_coord = 0
                y_coord += height_subimage+size_line
            image_legend[y_coord:y_coord+height_subimage,
                         x_coord:x_coord+width_subimage,:] = subimage[:height_image,:width_image,:]
            x_coord += width_subimage+size_line
        self.image_legend = image_legend[0:y_coord+height_subimage,:,:]
        #cv2.imshow("legend",self.image_legend)
        #cv2.waitKey(0)

    def _generate_templates(self):
        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)),"labels_avdira")
        self.templates = []
        for size in range(8):
            templates_single_size = {}
            for letter in self.letters:
                if ord(letter) not in range(9,14):
                    imagename = os.path.join(dirname,"%d_%d.png"%(ord(letter),size))
                    image0 = cv2.imread(imagename,0)
                    height, width = image0.shape[:2]
                    image = np.ones((height+size*4,width),dtype=np.uint8)*255
                    image[size*2:size*2+height,:] = image0
                    templates_single_size[letter] = image
            self.templates.append(templates_single_size)


if __name__ == "__main__":

    def main_label_text_generator():
        import time
        import string

        time0 = time.time()
        label_text_generator = LabelGenerator(10)
        strings = ["Hi"*50,"hello","nice 2 meet u!@#$","12345"," Deep learning",string.printable]
        for c,_string in enumerate(strings):
            time0 = time.time()
            subimage_text = label_text_generator.make_text_label(characters=_string,
                                                                 size=3,
                                                                 _class=c)
            print(_string)
            print("processing time: %.3f ms"%((time.time()-time0)*1e3))
            cv2.imshow("image",subimage_text)
            key = cv2.waitKey(0)
            if key == ord("q"):
                quit()

    def main_label_generator():
        import time

        time0 = time.time()
        number_classes = 15
        label_generator = LabelGenerator(number_classes)
        print(time.time()-time0)

        size_image = 300
        image_origin = np.zeros((size_image,size_image,3),dtype=np.uint8)
        bbox_coord = (20,50,400,600)
        for _class in range(number_classes):
            time0 = time.time()
            image_labeled = label_generator.draw_bbox_with_text(image_origin,
                                                                text="Deep Learning is !!!",
                                                                size=5,
                                                                _class=_class,
                                                                bbox_coord=bbox_coord)
            print("total %03.2f ms"%((time.time()-time0)*1e3))
            cv2.imshow("image",image_labeled)
            key = cv2.waitKey(0)
            if key == ord("q"):
                quit()

    def main_label_legend_generator():
        import time

        time0 = time.time()
        number_classes = 15
        label_generator = LabelGenerator(number_classes)
        print(time.time()-time0)

        size_image = 200
        image_origin = np.zeros((size_image,size_image,3),dtype=np.uint8)
        label_generator.get_legend(size=5,
                                   names_class=[str(s)*10 for s in range(10)],
                                   height_image=size_image,
                                   width_image=size_image)
        bbox_coord = (20,50,400,600)
        for _class in range(number_classes):
            time0 = time.time()
            image_labeled = label_generator.draw_bbox_with_text(image_origin,
                                                                text="Deep Learning is !!!",
                                                                size=5,
                                                                _class=_class,
                                                                bbox_coord=bbox_coord)
            print("total %03.2f ms"%((time.time()-time0)*1e3))
            cv2.imshow("image",image_labeled)
            key = cv2.waitKey(0)
            if key == ord("q"):
                quit()
        image_labeled = label_generator.add_legend(image_labeled)
        cv2.imshow("image",image_labeled)
        key = cv2.waitKey(0)
        if key == ord("q"):
            quit()

    def main_label_shift_generator():
        import time

        time0 = time.time()
        number_classes = 15
        label_generator = LabelGenerator(number_classes)
        print(time.time()-time0)

        size_image = 600
        image_origin = np.zeros((size_image,size_image,3),dtype=np.uint8)
        label_generator.get_legend(size=5,
                                   names_class=[str(s)*10 for s in range(10)],
                                   height_image=size_image,
                                   width_image=size_image)
        bbox_coord = (10,130,300,700)
        for _class in range(number_classes):
            time0 = time.time()
            image_labeled = label_generator.draw_bbox_with_text(image_origin,
                                                                text=" Dep! ",
                                                                size=5,
                                                                _class=_class,
                                                                bbox_coord=bbox_coord)
            print("total %03.2f ms"%((time.time()-time0)*1e3))
            cv2.imshow("image",image_labeled)
            key = cv2.waitKey(0)
            if key == ord("q"):
                quit()
        image_labeled = label_generator.add_legend(image_labeled)
        cv2.imshow("image",image_labeled)
        key = cv2.waitKey(0)
        if key == ord("q"):
            quit()


    main_label_text_generator()
    #main_label_generator()
    #main_label_legend_generator()
    #main_label_shift_generator()
