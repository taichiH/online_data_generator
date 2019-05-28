#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image
from online_data_generator_msgs.msg import AnnotationInfo
from online_data_generator_msgs.srv import AnnotationTrigger, AnnotationTriggerResponse

class Trigger():

    def __init__(self):
        self.mask_img = None
        self.rgb_img = None

        self.mask_pub = rospy.Publisher(
            '~mask_image', Image, queue_size=1)
        self.rgb_pub = rospy.Publisher(
            '~rgb_image', Image, queue_size=1)
        self.annotation_info_pub = rospy.Publisher(
            '~annotation_info', AnnotationInfo, queue_size=1)

        rospy.Service(
            '/annotation_trigger/set', AnnotationTrigger, self.service_callback)

        self.subscribe()

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_mask_img = message_filters.Subscriber(
            '~input_mask_img', Image, queue_size=1)
        sub_rgb_img = message_filters.Subscriber(
            '~input_rgb_img', Image, queue_size=1)

        self.subs = [sub_mask_img, sub_rgb_img]
        if rospy.get_param('~approximate_sync', True):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self.image_cb)

    def image_cb(self, mask_img, rgb_img):
        self.mask_img = mask_img
        self.rgb_img = rgb_img

    def service_callback(self, req):
        if self.mask_img == None or self.rgb_img == None:
            return AnnotationTriggerResponse(message='non image', status=False)

        mask_img_msg = self.mask_img
        mask_img_msg.header = self.mask_img.header

        rgb_img_msg = self.rgb_img
        rgb_img_msg.header = self.mask_img.header

        annotation_info_msg = AnnotationInfo()
        annotation_info_msg.header = self.mask_img.header
        annotation_info_msg.image_path = req.info.image_path
        annotation_info_msg.label = req.info.label
        annotation_info_msg.reset = req.info.reset

        self.mask_pub.publish(mask_img_msg)
        self.rgb_pub.publish(rgb_img_msg)
        self.annotation_info_pub.publish(annotation_info_msg)

        return AnnotationTriggerResponse(message='success', status=True)

if __name__=='__main__':
    rospy.init_node('annotation_trigger')
    trigger = Trigger()
    rospy.spin()
