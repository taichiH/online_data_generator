#ifndef _ONLINE_DATA_GENERATOR_DEEP_FLOW_H_
#define _ONLINE_DATA_GENERATOR_DEEP_FLOW_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

namespace online_data_generator
{
  class DeepFlow : public nodelet::Nodelet
  {
  public:

  protected:
    virtual void onInit();
    virtual void callback(const sensor_msgs::ImageConstPtr &img_msg);

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber img_sub_;
    ros::Publisher output_img_pub_;

    cv::Ptr<cv::DenseOpticalFlow> opt_ = cv::optflow::createOptFlow_DeepFlow();

    cv::Mat input_img_;
    cv::Mat current_img_;
    cv::Mat prev_img_;

    bool debug_ = false;
    float threshold_ = 0.5;
  private:
  };
} // end online_data_generator

#endif
