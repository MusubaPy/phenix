# CMake generated Testfile for 
# Source directory: /phenix_ws/mujoco_mpc/mjpc/test/tasks
# Build directory: /phenix_ws/mujoco_mpc/mjpc/test/tasks
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(TasksTest.Task "/phenix_ws/mujoco_mpc/bin/task_test" "--gtest_filter=TasksTest.Task")
set_tests_properties(TasksTest.Task PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;0;")
add_test(StepAllTasksTest.Task "/phenix_ws/mujoco_mpc/bin/task_test" "--gtest_filter=StepAllTasksTest.Task")
set_tests_properties(StepAllTasksTest.Task PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;0;")
add_test(QuadrupedStabilityTest.AutomaticGaitMaintainsPose "/phenix_ws/mujoco_mpc/bin/quadruped_stability_test" "--gtest_filter=QuadrupedStabilityTest.AutomaticGaitMaintainsPose")
set_tests_properties(QuadrupedStabilityTest.AutomaticGaitMaintainsPose PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;18;test;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;0;")
add_test(QuadrupedStabilityTest.ManualTrotRemainsBounded "/phenix_ws/mujoco_mpc/bin/quadruped_stability_test" "--gtest_filter=QuadrupedStabilityTest.ManualTrotRemainsBounded")
set_tests_properties(QuadrupedStabilityTest.ManualTrotRemainsBounded PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;18;test;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;0;")
add_test(QuadrupedStabilityTest.AutomaticGaitDebugWeights "/phenix_ws/mujoco_mpc/bin/quadruped_stability_test" "--gtest_filter=QuadrupedStabilityTest.AutomaticGaitDebugWeights")
set_tests_properties(QuadrupedStabilityTest.AutomaticGaitDebugWeights PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;18;test;/phenix_ws/mujoco_mpc/mjpc/test/tasks/CMakeLists.txt;0;")
