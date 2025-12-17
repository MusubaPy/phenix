# CMake generated Testfile for 
# Source directory: /phenix_ws/mujoco_mpc/mjpc/test/planners/robust
# Build directory: /phenix_ws/mujoco_mpc/mjpc/test/planners/robust
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(RobustPlannerTest.RandomSearch "/phenix_ws/mujoco_mpc/bin/robust_planner_test" "--gtest_filter=RobustPlannerTest.RandomSearch")
set_tests_properties(RobustPlannerTest.RandomSearch PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/planners/robust/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/planners/robust/CMakeLists.txt;0;")
