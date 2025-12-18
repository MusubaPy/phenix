# CMake generated Testfile for 
# Source directory: /phenix_ws/mujoco_mpc/mjpc/test/estimator
# Build directory: /phenix_ws/mujoco_mpc/mjpc/test/estimator
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(BatchFilter.Box3Drot "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/batch_filter_test" "--gtest_filter=BatchFilter.Box3Drot")
set_tests_properties(BatchFilter.Box3Drot PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;15;test;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(PriorCost.Particle "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/batch_prior_test" "--gtest_filter=PriorCost.Particle")
set_tests_properties(PriorCost.Particle PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;18;test;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(PriorCost.Box "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/batch_prior_test" "--gtest_filter=PriorCost.Box")
set_tests_properties(PriorCost.Box PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;18;test;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(Estimator.Kalman "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/kalman_test" "--gtest_filter=Estimator.Kalman")
set_tests_properties(Estimator.Kalman PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;21;test;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(Unscented.Particle1D "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unscented_test" "--gtest_filter=Unscented.Particle1D")
set_tests_properties(Unscented.Particle1D PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;24;test;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
add_test(Unscented.Box3Drot "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unscented_test" "--gtest_filter=Unscented.Box3Drot")
set_tests_properties(Unscented.Box3Drot PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" WORKING_DIRECTORY "/phenix_ws/mujoco_mpc/mjpc/test" _BACKTRACE_TRIPLES "/usr/share/cmake-3.28/Modules/GoogleTest.cmake;402;add_test;/phenix_ws/mujoco_mpc/mjpc/test/CMakeLists.txt;31;gtest_add_tests;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;24;test;/phenix_ws/mujoco_mpc/mjpc/test/estimator/CMakeLists.txt;0;")
