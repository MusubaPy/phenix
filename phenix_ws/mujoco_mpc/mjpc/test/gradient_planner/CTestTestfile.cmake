# CMake generated Testfile for 
# Source directory: /phenix_ws/mujoco_mpc/mjpc/test/gradient_planner
# Build directory: /phenix_ws/mujoco_mpc/mjpc/test/gradient_planner
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(GradientPlannerTest.Particle "/phenix_ws/mujoco_mpc/bin/gradient_planner_test" "--gtest_filter=GradientPlannerTest.Particle")
set_tests_properties(GradientPlannerTest.Particle PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;40;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.Gradient "/phenix_ws/mujoco_mpc/bin/gradient_test" "--gtest_filter=GradientTest.Gradient")
set_tests_properties(GradientTest.Gradient PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;40;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;18;test;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.ZeroTest "/phenix_ws/mujoco_mpc/bin/zero_test" "--gtest_filter=GradientTest.ZeroTest")
set_tests_properties(GradientTest.ZeroTest PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;40;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;25;test;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.LinearTest "/phenix_ws/mujoco_mpc/bin/linear_test" "--gtest_filter=GradientTest.LinearTest")
set_tests_properties(GradientTest.LinearTest PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;40;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;28;test;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
add_test(GradientTest.CubicTest "/phenix_ws/mujoco_mpc/bin/cubic_test" "--gtest_filter=GradientTest.CubicTest")
set_tests_properties(GradientTest.CubicTest PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;40;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;31;test;/phenix_ws/mujoco_mpc/mjpc/test/gradient_planner/CMakeLists.txt;0;")
