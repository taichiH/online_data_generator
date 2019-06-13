#!/usr/bin/env python

import time
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from online_data_generator_msgs.srv import GetImage, GetImageResponse
from std_srvs.srv import Trigger, TriggerResponse

class FlowMaskExtractor():

    def __init__(self):
        self.pub = rospy.Publisher('~output', Image, queue_size=1)
        self.get_flow_mask =rospy.ServiceProxy(
            '/deep_flow/get_deep_flow_mask', GetImage)
        rospy.Service(
            '~extract', Trigger, self.trigger_callback)

    def trigger_callback(self, req):
        header = Header()
        header.stamp = rospy.Time.now()
        get_flow_mask_res = self.get_flow_mask(return_image=False, publish_time=10.0, header=header)
        if get_flow_mask_res.return_image:
            self.pub.publish(get_flow_mask_res.output_image)
            res = TriggerResponse(success=True)

        res = TriggerResponse(success=True)
        return res

if __name__=='__main__':
    rospy.init_node('flow_mask_extractor')
    flow_mask_extractor = FlowMaskExtractor()
    rospy.spin()
