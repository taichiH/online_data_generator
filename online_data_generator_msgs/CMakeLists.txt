cmake_minimum_required(VERSION 2.8.3)
project(online_data_generator_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  geometry_msgs
  sensor_msgs
  std_msgs
  )

add_message_files(
  FILES
  AnnotationInfo.msg
  )

add_service_files(
  FILES
  AnnotationTrigger.srv
  Segmentation.srv
  GetImage.srv
  ConvertImage.srv
  )

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
  )

catkin_package()
