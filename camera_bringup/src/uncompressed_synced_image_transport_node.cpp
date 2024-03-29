// Copyright 2021, Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/highgui.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    std::cout << "Made it here" << std::endl;
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(1);
  } catch (const cv_bridge::Exception & e) {
    auto logger = rclcpp::get_logger("my_subscriber");
    RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

void leftCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    std::cout << "Made it left here" << std::endl;
    cv::imshow("leftView", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(1);
  } catch (const cv_bridge::Exception & e) {
    auto logger = rclcpp::get_logger("my_subscriber");
    RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

void rightCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    std::cout << "Made it right here" << std::endl;
    cv::imshow("rightView", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(1);
  } catch (const cv_bridge::Exception & e) {
    auto logger = rclcpp::get_logger("my_subscriber");
    RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("image_listener", options);
  cv::namedWindow("view");
  cv::startWindowThread();
  image_transport::ImageTransport it(node);
  image_transport::Subscriber sub_color = it.subscribe("/repub_realsense_color", 10, imageCallback);
  image_transport::Subscriber sub_left = it.subscribe("/repub_left", 10, leftCallback);
  image_transport::Subscriber sub_right = it.subscribe("/repub_right", 10, rightCallback);
  rclcpp::spin(node);
  cv::destroyWindow("view");
  cv::destroyWindow("leftView");
  cv::destroyWindow("rightView");

  return 0;
}
