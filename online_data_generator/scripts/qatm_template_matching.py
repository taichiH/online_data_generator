#!/usr/bin/env python

import rospy
import rospkg
import cv_bridge
from sensor_msgs.msg import Image

from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *

# +
# import functions and classes from qatm_pytorch.py

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
# -



class QATM():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.use_cuda = rospy.get_param('~use_cuda', True)
        self.templates_dir = rospy.get_param('~templates_dir', os.path.join(pkg_dir,'templates'))
        self.thresholds = rospy.get_param('~thresholds', os.path.join(pkg_dir, 'data/thresholds.csv'))
        self.alpha = rospy.get_param('~alpha', 25)

        rospy.loginfo("define model...")
        self.model = CreateModel(
            model=models.vgg19(pretrained=True).features, alpha=self.alpha, use_cuda=self.use_cuda)

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
            Path(self.templates_dir), raw_image, thresh_csv=self.thresholds, image_name='input')

        rospy.loginfo("calculate score...")
        scores, w_array, h_array, thresh_list = run_multi_sample(self.model, dataset)

        rospy.loginfo("nms...")
        boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
        output_image = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name=None)

        rospy.loginfo("publish image")
        msg_viz = cv_bridge.CvBridge().cv2_to_imgmsg(output_image, encoding='bgr8')
        msg_viz.header = imgmsg.header
        self.pub_viz.publish(msg_viz)


if __name__ == '__main__':
    rospy.init_node('qatm_template_matching')
    qatm = QATM()
    rospy.spin()
