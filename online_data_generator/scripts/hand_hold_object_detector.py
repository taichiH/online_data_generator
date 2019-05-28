#!/usr/bin/env python

import cv2
import numpy as np
import copy
import sys

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import RectArray, Rect

class HandHoldObjectDetector():

    def __init__(self):
        self.cv_bridge = CvBridge()
        self.area_size = rospy.get_param('~area_size', 20)
        self.depth_thresh = rospy.get_param('~depth_thresh', 100) # mm

        self.debug = rospy.get_param('~debug', True)
        self.debug_img_pub = rospy.Publisher(
            '~output/debug', Image, queue_size=1)
        self.mask_pub = rospy.Publisher(
            '~output/mask', Image, queue_size=1)
        self.rgb_pub = rospy.Publisher(
            '~output/rgb', Image, queue_size=1)

        self.subscribe()

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)

        # sub_depth_img = message_filters.Subscriber(
        #     '~input_depth_img', Image, queue_size=1)
        # sub_rgb_img = message_filters.Subscriber(
        #     '~input_rgb_img', Image, queue_size=1)
        # sub_hand_rect = message_filters.Subscriber(
        #     '~input_hand_rect', RectArray, queue_size=1)

        sub_depth_img = message_filters.Subscriber(
            '/camera/depth_registered/hw_registered/image_rect', Image, queue_size=1000)
        sub_rgb_img = message_filters.Subscriber(
            '/camera/rgb/image_rect_color', Image, queue_size=1000)
        sub_hand_rect = message_filters.Subscriber(
            '/ssd_object_detector/output/rect', RectArray, queue_size=1000)

        self.subs = [sub_depth_img, sub_rgb_img, sub_hand_rect]
        if rospy.get_param('~approximate_sync', True):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self.callback)

    def callback(self, depth_img_msg, rgb_img_msg, hand_rect_msg):
        if len(hand_rect_msg.rects) == 0:
            return

        depth = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, 'passthrough')
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_img_msg, 'bgr8')
        debug_img = copy.copy(rgb)
        rgb_output = copy.copy(rgb)

        depth.flags.writeable = True
        depth[np.isnan(depth)] = 0

        hand_depth = sys.maxsize
        for rect in hand_rect_msg.rects:
            c_x, c_y =  int(rect.x + (rect.width * 0.5)), int(rect.y + (rect.height * 0.5))

            depth_ave = depth[c_y - self.area_size : c_y + self.area_size,
                              c_x - self.area_size : c_x + self.area_size].mean()
            if depth_ave < hand_depth:
                hand_depth = depth_ave

        mask = np.full((rgb.shape[0], rgb.shape[1]), 255, dtype=np.uint8)
        mask[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] = 0
        mask[depth>hand_depth + self.depth_thresh] = 0
        mask[depth<hand_depth - self.depth_thresh] = 0

        rgb_output[:,:,:][mask==255] = [0, 0 ,255]

        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_output, "bgr8")
        rgb_msg.header = rgb_img_msg.header
        self.rgb_pub.publish(rgb_msg)

        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask, "mono8")
        mask_msg.header = rgb_img_msg.header
        self.mask_pub.publish(mask_msg)

        if self.debug:
            # print(hand_depth.mean())
            # print('top_left', c_y - self.area_size, c_x - self.area_size)
            # print('bottom_right', c_y + self.area_size, c_x + self.area_size)
            debug_img[c_y - self.area_size : c_y + self.area_size, c_x - self.area_size : c_x + self.area_size,:] = [255,0,0]
            debug_img = cv2.circle(debug_img, (c_x,c_y), 5, (0,0,255), -1)
            debug_image_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_image_msg.header = rgb_img_msg.header
            self.debug_img_pub.publish(debug_image_msg)



if __name__=='__main__':
    rospy.init_node('hand_hold_object_detector')
    hhod = HandHoldObjectDetector()
    rospy.spin()
