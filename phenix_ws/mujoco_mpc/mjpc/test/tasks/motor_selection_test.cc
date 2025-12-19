#include "mjpc/tasks/quadruped_mod/quadruped_mod.h"
#include <gtest/gtest.h>

using mjpc::QuadrupedFlatMod;

TEST(MotorSelectionTest, DegenerateNormal) {
  double contact_normal[3] = {0.0, 0.0, 0.0};
  double motors[2][3] = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}};
  double plane_normal[3] = {0.0, 1.0, 0.0};

  auto res = QuadrupedFlatMod::ResidualFn::SelectMotorUsingNormalForTest(
      contact_normal, motors, plane_normal);
  EXPECT_FALSE(res.valid);
}

TEST(MotorSelectionTest, DiscreteChoice) {
  double contact_normal[3] = {0.0, 0.0, 1.0};
  double motors[2][3] = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}};
  double plane_normal[3] = {0.0, 1.0, 0.0};

  auto res = QuadrupedFlatMod::ResidualFn::SelectMotorUsingNormalForTest(
      contact_normal, motors, plane_normal);
  EXPECT_TRUE(res.valid);
  EXPECT_EQ(res.index, 0);
  EXPECT_NEAR(res.angle, 0.0, 1e-6);
}

TEST(MotorSelectionTest, SoftBlend) {
  double contact_normal[3] = {1.0, 0.0, 1.0};
  double norm = std::sqrt(2.0);
  contact_normal[0] /= norm; contact_normal[2] /= norm;
  double motors[2][3] = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}};
  double plane_normal[3] = {0.0, 1.0, 0.0};

  auto res_blend = QuadrupedFlatMod::ResidualFn::SelectMotorUsingNormalForTest(
      contact_normal, motors, plane_normal, /*blend_beta=*/1.0);
  EXPECT_TRUE(res_blend.valid);
  EXPECT_EQ(res_blend.index, -1);
  // blended projection should have non-zero components in x and z
  EXPECT_GT(std::abs(res_blend.motor_proj[0]), 1e-9);
  EXPECT_GT(std::abs(res_blend.motor_proj[2]), 1e-9);
}
