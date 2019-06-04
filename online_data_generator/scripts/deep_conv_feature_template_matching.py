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
        self.template_dir = rospy.get_param('template_dir')

        self.template_loaded = False
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


    def load_templates(self):
        templates = []
        template_files = os.listdir(self.template_dir)
        for template_file in template_files:
            template_path = os.path.joint(self.template_dir, template_file)
            raw_template = cv2.imread(template_path)[..., ::-1]
            feature_map, extraction_layer = self.create_template_feature_map(raw_template)
            templates.append((feature_map, extraction_layer))
        return templates

    def create_template_feature_map(self, raw_template, fe):
        template = image_transform(raw_template.copy()).unsqueeze(0)
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

        return fe.template_feature_map, extraction_layer

    def load_image(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        try:
            self.image = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        except:
            rospy.logerr('failed transform image')
            return
        raw_image = copy.copy(self.image)
        return raw_image

    def create_image_feature_map(self, raw_image, fe, extraction_layer):
        image = image_transform(raw_image.copy()).unsqueeze(0)
        image_handle = fe.model[fe.index[extraction_layer]].register_forward_hook(
            fe.save_image_feature_map)
        fe.model(image)
        image_handle.remove()

        if self.use_cuda:
            fe.image_feature_map = fe.image_feature_map.cpu()
        M = fe.image_feature_map.numpy()[0].astype(np.float32)
        return M

    def callback(self, imgmsg):
        vgg_feature = models.vgg13(pretrained=True).features
        fe = FeatureExtractor(
            vgg_feature, use_cuda=self.use_cuda, padding=True, threshold=self.threshold)

        if not self.tepmlate_loaded:
            self.templates = self.load_templates()

        raw_image = load_image(imgmsg)
        output_image = raw_image.astype(np.uint8).copy()

        for template, extraction_layer in self.templates:
            M = create_image_feature_map(raw_image, fe, extraction_layer)
            boxes, centers, scores, match_indices = fe.extract(
                template, M, extraction_layer, use_cython=False)

            nms_res = fe.nms(np.array(boxes), np.array(scores), thresh=0.5)
            print("detected objects: {}".format(len(nms_res)))
            for i in nms_res:
                output_image = cv2.rectangle(output_image, boxes[i][0], boxes[i][1], (255, 0, 0), 3)
                output_image = cv2.circle(output_image, centers[i], 3, (0, 0, 255), 1)
                for match_index in match_indices:
                    i, j = match_index
                    output_image = cv2.circle(output_image, (i, j), 5, (0, 255, 0), 1)

        msg_viz = bridge.cv2_to_imgmsg(output_image, encoding='rgb8')
        msg_viz.header = imgmsg.header
        self.pub_viz.publish(msg_viz)


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
