cmake_minimum_required(VERSION 3.8)
project(camera_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(lifelong_msgs REQUIRED)
#find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
#find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS highgui imgcodecs imgproc videoio)
find_package(pluginlib REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include_directories(include)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(uncompressed_realsense_image_transport_node src/uncompressed_realsense_image_transport_node.cpp)
target_link_libraries(uncompressed_realsense_image_transport_node
  cv_bridge::cv_bridge
  image_transport::image_transport
  opencv_highgui
  rclcpp::rclcpp
)

add_executable(uncompressed_synced_image_transport_node src/uncompressed_synced_image_transport_node.cpp)
target_link_libraries(uncompressed_synced_image_transport_node
  cv_bridge::cv_bridge
  image_transport::image_transport
  opencv_highgui
  rclcpp::rclcpp
)

add_executable(uncompressed_zed_realsense_image_transport_node src/uncompressed_zed_realsense_image_transport_node.cpp)
target_link_libraries(uncompressed_zed_realsense_image_transport_node
  cv_bridge::cv_bridge
  image_transport::image_transport
  opencv_highgui
  rclcpp::rclcpp
)

add_executable(manual_splat_collection_node src/manual_splat_collection_node.cpp)
target_link_libraries(manual_splat_collection_node
  cv_bridge::cv_bridge
  image_transport::image_transport
  opencv_highgui
  rclcpp::rclcpp
)

install(
  TARGETS uncompressed_zed_realsense_image_transport_node uncompressed_synced_image_transport_node manual_splat_collection_node uncompressed_realsense_image_transport_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/image_visualization.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/collect_images_for_nerf.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/manual_splat_collection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/manual_realsense_splat_collection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/img_depth_img_folder_to_rostopic.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/folder_image_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/manual_left_zed_realsense_splat_collection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/enter_key_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/color_image_compression_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/arducam_right_compressed_converter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/get_aruco_pose_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/nerf_collection_pose.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/realsense_compressed_converter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/realsense_uncompression_compression.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/new_realsense_compression_uncompression.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/compress_realsense_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/arducam_left_compressed_converter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/image_pose_filtering.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/4_arducam_compressed_converter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/aruco_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/calibrate_extrinsics_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/uncompressed_synced_image_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/aruco_pose_calc_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/find_relative_cam_poses.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/save_images_for_calibration.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/save_images_for_calibration_vid_stream.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/compare_arucos.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/3_arducam_compressed_converter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/3_arducam_compressed_converter_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/view_4_cam.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
