#!/usr/bin/env python

import copy
import argparse
import matplotlib.pyplot as plt

import cv2
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import models, transforms, utils
import numpy as np

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from online_data_generator_msgs.srv import Segmentation, SegmentationResponse
from feature_extractor import FeatureExtractor


class DeepConvFeatureTemplateMatching():

    def __init__(self):
        self.use_cuda = rospy.get_param('use_cuda', True)
        self.threshold = rospy.get_param('threshold', 0.5)

        self.image = None

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )])

        self.pub_viz = rospy.Publisher('~output', Image, queue_size=1)
        rospy.Service('~match', Segmentation, self.service_callback)
        rospy.Subscriber('~input', Image, self.callback, queue_size=1)

    def callback(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        try:
            self.image = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        except:
            rospy.logerr('failed transform image')
            return

        raw_image = copy.copy(self.image)
        image = image_transform(raw_image.copy()).unsqueeze(0)

        vgg_feature = models.vgg13(pretrained=True).features
        FE = FeatureExtractor(
            vgg_feature, use_cuda=args.use_cuda, padding=True, threshold=args.threshold)
        boxes, centers, scores, match_indices = FE(template, image, use_cython=False)

        d_img = raw_image.astype(np.uint8).copy()
        nms_res = nms(np.array(boxes), np.array(scores), thresh=0.5)

        print("detected objects: {}".format(len(nms_res)))

        for i in nms_res:
            d_img = cv2.rectangle(d_img, boxes[i][0], boxes[i][1], (255, 0, 0), 3)
            d_img = cv2.circle(d_img, centers[i], 3, (0, 0, 255), 1)
            for match_index in match_indices:
                i, j = match_index
                d_img = cv2.circle(d_img, (i, j), 5, (0, 255, 0), 1)


        msg_viz = bridge.cv2_to_imgmsg(d_img, encoding='rgb8')
        msg_viz.header = imgmsg.header
        self.pub_viz.publish(msg_viz)

    def nms(self, dets, scores, thresh):
        x1 = dets[:, 0, 0]
        y1 = dets[:, 0, 1]
        x2 = dets[:, 1, 0]
        y2 = dets[:, 1, 1]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


    # input: rgb image
    def service_callback(self, req):
        bridge = cv_bridge.CvBridge()
        try:
            tmpl_image = bridge.imgmsg_to_cv2(req.input_image, desired_encoding='bgr8')
            template = image_transform(raw_image.copy()).unsqueeze(0)
        except:
            rospy.logerr('failed transform tmpl')
            return

        res = SegmentationResponse()
        return res


if __name__ == '__main__':
    rospy.init_node('deep_conv_feature_template_matching')
    dcftm = DeepConvFeatureTemplateMatching()
    rospy.spin()



