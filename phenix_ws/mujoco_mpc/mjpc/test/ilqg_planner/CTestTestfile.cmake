# CMake generated Testfile for 
# Source directory: /phenix_ws/mujoco_mpc/mjpc/test/ilqg_planner
# Build directory: /phenix_ws/mujoco_mpc/mjpc/test/ilqg_planner
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(iLQGTest.Particle "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/ilqg_test" "--gtest_filter=iLQGTest.Particle")
set_tests_properties(iLQGTest.Particle PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/ilqg_planner/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/ilqg_planner/CMakeLists.txt;0;")
add_test(iLQGTest.BackwardPass "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/backward_pass_test" "--gtest_filter=iLQGTest.BackwardPass")
set_tests_properties(iLQGTest.BackwardPass PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/ilqg_planner/CMakeLists.txt;18;test;/phenix_ws/mujoco_mpc/mjpc/test/ilqg_planner/CMakeLists.txt;0;")
