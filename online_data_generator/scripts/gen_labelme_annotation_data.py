#!/usr/bin/env python

import io
import os
import cv2
import numpy as np
import PIL.Image
import traceback
import json
from copy import copy

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from online_data_generator_msgs.msg import AnnotationInfo

class AnnotationData():

    def __init__(self, image_path='', shapes=[], image_data=None,
                 line_color=[0,255,0,128], fill_color=[255,0,0,128]):

        self.data = {'imagePath':image_path,
                     'shapes':shapes,
                     'imageData': image_data,
                     'lineColor':line_color,
                     'fillColor':fill_color}

    def reset(self):
        self.data = {'imagePath':'',
                     'shapes':[],
                     'imageData': None,
                     'lineColor':[0,255,0,128],
                     'fillColor':[255,0,0,128]}

class GenLabelmeAnnotationData():

    def __init__(self):
        self.ins = 0
        self.image_num = 0
        self.label = 'sample_label'

        self.debug = True
        self.cv_bridge = CvBridge()
        self.debug_img_pub = rospy.Publisher(
            '~output', Image, queue_size=1)
        self.subscribe()

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)

        sub_mask_img = message_filters.Subscriber(
            '~input_mask_img', Image, queue_size=1)
        sub_rgb_img = message_filters.Subscriber(
            '~input_rgb_img', Image, queue_size=1)
        sub_annotation_info = message_filters.Subscriber(
            '~input_annotation_info', AnnotationInfo, queue_size=1)

        self.subs = [sub_mask_img, sub_rgb_img, sub_annotation_info]
        if rospy.get_param('~approximate_sync', True):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            print(2)
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)

        sync.registerCallback(self.callback)

    def callback(self, mask_img, rgb_img, annotation_info):
        print('callback')

        mask = self.cv_bridge.imgmsg_to_cv2(mask_img, 'mono8')
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_img, 'bgr8')
        self.label = annotation_info.label
        self.save_dir = annotation_info.image_path

        if mask is None:
            rospy.logwarn('input msg is empty')
            return

        debug_img = np.zeros(
            (mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(debug_img.shape[2]):
            debug_img[:,:,i][mask==255] = [200,200,200][i]

        annotation_data = self.create_json(debug_img, rgb)
        if not self.save_file(annotation_data.data, rgb):
            return

        self.image_num += 1

        if self.debug:
            debug_img_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, 'bgr8')
            debug_img_msg.header = mask_img.header
            self.debug_img_pub.publish(debug_img_msg)

    def create_json(self, debug_img, rgb_image):
        annotation_data = AnnotationData()
        annotation_data.reset()

        shape = {'line_color':None, 'points':[], 'fill_color':None, 'label':''}
        gray = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127,255, 0)
        dst, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 1:
            rospy.loginfo('%d instances exist' %(len(contours)))

        for ins, contour in enumerate(contours):
            for i in range(contours[ins].shape[0]):
                debug_img[contours[0][i,0,1], contours[0][i,0,0], :] = [255,0,0]
                shape['points'].append(
                    map(lambda x: float(x), [contours[0][i,0,0], contours[0][i,0,1]]))

            shape['label'] = self.label + '-' + str(ins)

        # annotation_data.data['imageData'] = self.convert_image(rgb_image)
        annotation_data.data['imageData'] = None
        annotation_data.data['imagePath'] = str(self.image_num).zfill(4) + '.jpg'
        annotation_data.data['shapes'].append(shape)
        return annotation_data

    def save_file(self, json_dict, output_image):
        image_num = str(self.image_num).zfill(4)

        if not cv2.imwrite(
                os.path.join(self.save_dir, image_num + '.jpg'),
                output_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100]):
            rospy.logerr(
                'failed save image: [%s]' %(save_path))
            return False

        try:
            with open(os.path.join(self.save_dir, image_num + '.json'), 'w') as outfile:
                json.dump(json_dict, outfile, ensure_ascii=False, indent=2)
        except:
            rospy.logerr(
                'failed save json file: [%s]' %(image_num + '.json'))
            traceback.print_exc()
            return False

        return True

    def convert_image(self, np_img):
        img = PIL.Image.fromarray(np_img)
        with io.BytesIO() as imgBytesIO:
            img.save(imgBytesIO, "JPEG")
            imgBytesIO.seek(0)
            data = imgBytesIO.read()
        return data

if __name__=='__main__':
    rospy.init_node('gen_labelme_annotation_data')
    glad = GenLabelmeAnnotationData()
    rospy.spin()
