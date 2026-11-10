cmake_minimum_required(VERSION 3.8)
project(estun_s3_60_g2_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  extraction
  mesh
  source
  urdf
  validation
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/generate_description.py
  scripts/generate_gripper_meshes.py
  scripts/generate_link_meshes.py
  scripts/generate_model_manifest.py
  scripts/validate_kinematics.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_kinematic_model test/test_kinematic_model.py)
endif()

ament_package()
