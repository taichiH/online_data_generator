#include "online_data_generator/deep_flow.h"

namespace online_data_generator
{
  void DeepFlow::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();

    output_img_pub_ = pnh_.advertise<sensor_msgs::Image>("output", 1);
    img_sub_ = pnh_.subscribe("input", 1, &DeepFlow::callback, this);
  }

  void DeepFlow::callback(const sensor_msgs::ImageConstPtr &img_msg){
    cv_bridge::CvImagePtr cv_rgb;
    try {
      cv_rgb = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e){
      NODELET_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    input_img_ = cv_rgb->image;
    cv::cvtColor(input_img_, current_img_, cv::COLOR_BGR2GRAY);

    if (prev_img_.empty())
      current_img_.copyTo(prev_img_);

    cv::Mat flow_img;
    // output flow_img has 2 channels
    opt_->calc(current_img_, prev_img_, flow_img);

    cv::Mat flow_vec[2];
    cv::split(flow_img, flow_vec);

    cv::Mat magnitude, normalized_magnitude, angle;
    cv::cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    cv::normalize(magnitude, normalized_magnitude, 0.0, 1.0, cv::NORM_MINMAX);

    cv::Mat hsv_planes[3];
    hsv_planes[0] = angle;
    hsv_planes[1] = normalized_magnitude;
    hsv_planes[2] = cv::Mat::ones(magnitude.size(), CV_32F);

    cv::Mat hsv, buf, output_img;
    cv::merge(hsv_planes, 3, hsv);
    cv::cvtColor(hsv, buf, cv::COLOR_HSV2BGR);
    buf *= 255.0;
    buf.convertTo(output_img, CV_8UC3);
    current_img_.copyTo(prev_img_);

    output_img_pub_.publish(cv_bridge::CvImage(img_msg->header,
                                               sensor_msgs::image_encodings::BGR8,
                                               output_img).toImageMsg());
  }
} // online_data_generator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(online_data_generator::DeepFlow, nodelet::Nodelet)
