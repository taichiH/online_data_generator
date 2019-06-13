#!/usr/bin/env python

'''
MIT License

Copyright (c) 2019 Hiromichi Kamata

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import rospy
import rospkg
import cv_bridge
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import LabelArray, Label
from jsk_recognition_msgs.msg import RectArray, Rect

from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *

import ast
import types
import sys

pkg_dir = rospkg.RosPack().get_path('online_data_generator')
qatm_pytorch = os.path.join(pkg_dir, 'scripts/qatm_pytorch.py')

print("import qatm_pytorch.py...")
with open(qatm_pytorch) as f:
       p = ast.parse(f.read())

for node in p.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        p.body.remove(node)

module = types.ModuleType("mod")
code = compile(p, "mod.py", 'exec')
sys.modules["mod"] = module
exec(code,  module.__dict__)

from mod import *

class QATM():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.use_cuda = rospy.get_param('~use_cuda', True)
        self.templates = rospy.get_param('~templates', os.path.join(pkg_dir, 'data/thresholds.csv'))
        self.alpha = rospy.get_param('~alpha', 25)

        rospy.loginfo("define model...")
        self.model = CreateModel(
            model=models.vgg19(pretrained=True).features, alpha=self.alpha, use_cuda=self.use_cuda)

        self.labels_pub = rospy.Publisher('~output/labels', LabelArray, queue_size=1)
        self.rects_pub = rospy.Publisher('~output/rects', RectArray, queue_size=1)
        self.pub_viz = rospy.Publisher('~output', Image, queue_size=1)

        rospy.Subscriber('~input', Image, self.callback, queue_size=1)


    def callback(self, imgmsg):
        raw_image = None
        try:
            raw_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        except:
            rospy.logerr('failed transform image')
            return

        dataset = ImageDataset(
            pkg_dir, raw_image, thresh_csv=self.templates, image_name='input')

        rospy.loginfo("calculate score...")
        scores, w_array, h_array, thresh_list, label_list = run_multi_sample(self.model, dataset)

        rospy.loginfo("nms...")
        boxes, indices = nms_multi(scores, w_array, h_array, thresh_list, label_list)
        output_image = plot_result_multi(dataset.image_raw, boxes, label_list, indices, show=False, save_name=None)

        labels_msg = LabelArray()
        rects_msg = RectArray()
        for i in range(len(indices)):
            rect_msg = Rect()
            label_msg = Label()
            box = boxes[i][None, :,:][0]
            rect_msg.x = box[0][0]
            rect_msg.y = box[0][1]
            rect_msg.width = box[1][0] - box[0][0]
            rect_msg.height = box[1][1] - box[0][1]
            label_msg.name = label_list[indices[i]]
            rects_msg.rects.append(rect_msg)
            labels_msg.labels.append(label_msg)

        self.labels_pub.publish(labels_msg)
        self.rects_pub.publish(rects_msg)

        msg_viz = cv_bridge.CvBridge().cv2_to_imgmsg(output_image, encoding='bgr8')
        msg_viz.header = imgmsg.header
        self.pub_viz.publish(msg_viz)


if __name__ == '__main__':
    rospy.init_node('qatm_template_matching')
    qatm = QATM()
    rospy.spin()
