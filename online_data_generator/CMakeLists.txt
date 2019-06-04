cmake_minimum_required(VERSION 2.8.3)
project(online_data_generator)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  message_filters
  tf
  online_data_generator_msgs
  nodelet
)

find_package(OpenCV 3 REQUIRED)

set(CMAKE_CXX_FLAGS "--std=c++11")
message("OPENCV ${OpenCV_VERSION} FOUND")

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  )
link_directories(${OPENCV_LIBRARY_DIRS})

catkin_package(
  CATKIN_DEPENDS online_data_generator_msgs
  LIBRARIES ${PROJECT_NAME}
  INCLUDE_DIRS include
  )

add_library(${PROJECT_NAME} SHARED src/deep_flow.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
