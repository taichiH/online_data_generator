#!/usr/bin/env python

from sensor_msgs.msg import Image
from online_data_generator.srv import Segmentation


if __name__=='__main__':
    rospy.init_node('image_service_publisher')
    rospy.
    client =rospy.ServiceProxy(
        '/deep_conv_feature_template_matching/match', Segmentation)
    res = client(Segmentation(input_image=))
