#!/usr/bin/env python

import os
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
        self.use_cuda = rospy.get_param('~use_cuda', True)
        self.threshold = rospy.get_param('~threshold', 0.5)
        self.template_dir = rospy.get_param('~template_dir', '/home/taichi/annotation_ws/src/online_data_generator/online_data_generator/templates')

        self.template_loaded = False
        self.image = None

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )])

        vgg_feature = models.vgg13(pretrained=True).features
        self.fe = FeatureExtractor(
            vgg_feature, use_cuda=self.use_cuda, padding=True, threshold=self.threshold)

        self.pub_viz = rospy.Publisher('~output', Image, queue_size=1)
        self.pub_debug = rospy.Publisher('~debug', Image, queue_size=1)
        rospy.Service('~match', Segmentation, self.service_callback)
        rospy.Subscriber('~input', Image, self.callback, queue_size=1)

    def load_templates(self, fe):
        templates = []
        template_files = os.listdir(self.template_dir)
        for template_file in template_files:
            template_path = os.path.join(self.template_dir, template_file)
            raw_template = cv2.imread(template_path)[..., ::-1]
            feature_map, extraction_layer, template = self.create_template_feature_map(raw_template, fe)
            templates.append((template, feature_map, extraction_layer))
        return templates

    def create_template_feature_map(self, raw_template, fe):
        template = self.image_transform(raw_template.copy()).unsqueeze(0)
        if self.use_cuda:
            template = template.cuda()

        extraction_layer = fe.calc_extraction_layer(template)
        template_handle = fe.model[fe.index[extraction_layer]].register_forward_hook(
            fe.save_template_feature_map)
        fe.model(template)
        template_handle.remove()

        if self.use_cuda:
            fe.template_feature_map = fe.template_feature_map.cpu()

        self.template_loaded = True

        return fe.template_feature_map, extraction_layer, template

    def load_image(self, imgmsg):
        print('load_image')
        bridge = cv_bridge.CvBridge()
        try:
            raw_image = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        except:
            rospy.logerr('failed transform image')
            return
        # raw_image = copy.copy(self.image)
        return raw_image

    def create_image_feature_map(self, raw_image, fe, extraction_layer):
        image = self.image_transform(raw_image.copy()).unsqueeze(0)
        image_handle = fe.model[fe.index[extraction_layer]].register_forward_hook(
            fe.save_image_feature_map)
        fe.model(image)
        image_handle.remove()

        if self.use_cuda:
            fe.image_feature_map = fe.image_feature_map.cpu()

        M = fe.image_feature_map.numpy()[0].astype(np.float32)
        return M, image

    def callback(self, imgmsg):
        fe = self.fe
        # if not self.template_loaded:
        #     self.templates = self.load_templates(fe)

        self.templates = self.load_templates(fe)
        raw_image = self.load_image(imgmsg)
        output_image = raw_image.copy()

        msg_debug = cv_bridge.CvBridge().cv2_to_imgmsg(raw_image, encoding='bgr8')
        msg_debug.header = imgmsg.header
        self.pub_debug.publish(msg_debug)

        raw_image = raw_image[..., ::-1]

        for template, feature_map, extraction_layer in self.templates:
            M, image = self.create_image_feature_map(raw_image, fe, extraction_layer)
            boxes, centers, scores, match_indices = fe.extract(
                template, image, extraction_layer, use_cython=False)

            nms_res = fe.nms(np.array(boxes), np.array(scores), thresh=0.5)
            for i in nms_res:
                output_image = cv2.rectangle(output_image, boxes[i][0], boxes[i][1], (255, 0, 0), 3)
                output_image = cv2.circle(output_image, centers[i], int((boxes[i][1][0] - boxes[i][0][0])*0.2), (0, 0, 255), 2)

        msg_viz = cv_bridge.CvBridge().cv2_to_imgmsg(output_image, encoding='bgr8')
        msg_viz.header = imgmsg.header
        self.pub_viz.publish(msg_viz)

    def service_callback(self, req):
        bridge = cv_bridge.CvBridge()
        try:
            tmpl_image = bridge.imgmsg_to_cv2(req.input_image, desired_encoding='bgr8')
            template = self.image_transform(raw_image.copy()).unsqueeze(0)
        except:
            rospy.logerr('failed transform tmpl')
            return

        res = SegmentationResponse()
        return res


if __name__ == '__main__':
    rospy.init_node('deep_conv_feature_template_matching')
    dcftm = DeepConvFeatureTemplateMatching()
    rospy.spin()
