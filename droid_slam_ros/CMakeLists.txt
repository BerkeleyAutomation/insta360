cmake_minimum_required(VERSION 3.8)
project(droid_slam_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/droid_slam_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/space_bundle_adjust_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/view_multi_compressed_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/multi_ptcloud_d455_2zed_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/ptcloud_d435_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/online_multi_ptcloud_d455_2zed_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/multi_ptcloud_d455_zed_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/convert_d455_corrected_zed_folder_to_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/ptcloud_zed_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/convert_d455_2_zed_folder_to_bag.py
  DESTINATION lib/${PROJECT_NAME}
)


install(PROGRAMS
  scripts/convert_d455_folder_to_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/convert_zed_folder_to_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/view_compressed_bag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dummy_droid_slam_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/droid_slam_node_og.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/multi_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/multi_ptcloud_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)


install(PROGRAMS
  scripts/ptcloud_d455_global_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/karim_ptcloud_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/decode_image_poses_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dummy_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/ptcloud_real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/real_droid_subscriber_node_prime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/droid_slam_node_og_modified.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dummy_ros_image_droid_slam_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dummy_droid_subscriber_node_full_img.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dummy_droid_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dummy_droid_subscriber_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dynamic_frame_broadcaster_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  droid_slam
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  params
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  droid.pth
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
