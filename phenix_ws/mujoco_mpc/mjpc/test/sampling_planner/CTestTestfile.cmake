# CMake generated Testfile for 
# Source directory: /phenix_ws/mujoco_mpc/mjpc/test/sampling_planner
# Build directory: /phenix_ws/mujoco_mpc/mjpc/test/sampling_planner
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SamplingPlannerTest.RandomSearch "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/sampling_planner_test" "--gtest_filter=SamplingPlannerTest.RandomSearch")
set_tests_properties(SamplingPlannerTest.RandomSearch PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/sampling_planner/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/sampling_planner/CMakeLists.txt;0;")
