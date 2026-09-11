cmake_minimum_required(VERSION 3.8)
project(s3_60_eco_2024_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/load_track1_items.py
  scripts/check_reachability.py
  scripts/workspace_map.py
  scripts/plan_execute_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
