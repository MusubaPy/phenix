# CMake generated Testfile for 
# Source directory: /phenix_ws/mujoco_mpc/mjpc/test/state
# Build directory: /phenix_ws/mujoco_mpc/mjpc/test/state
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(StateTest.State "/phenix_ws/mujoco_mpc/bin/state_test" "--gtest_filter=StateTest.State")
set_tests_properties(StateTest.State PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/state/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/state/CMakeLists.txt;0;")
