cmake_minimum_required(VERSION 3.8)
project(s3_60_eco_2024)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  meshes
  launch
  rviz
  config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/demo_trajectory.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
