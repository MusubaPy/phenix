#include "gtest/gtest.h"
#include "mjpc/tasks/quadruped_mod/quadruped_mod.h"

using namespace mjpc;

TEST(PerFootResidualTest, NonDegenerateHindFoot) {
  double hip[3] = {0.0, 0.2, 0.0};
  double knee[3] = {0.0, -0.2, 0.0};
  double foot[3] = {0.1, 0.0, -0.25};
  double normal[3] = {0.0, 0.0, 1.0};
  double force[3] = {0.0, 0.0, 80.0};
  double out[3] = {0.0, 0.0, 0.0};
  bool ok = QuadrupedFlatMod::ResidualFn::ComputePerFootResidualForTest(
      force, normal, hip, knee, foot, 0.05, out);
  EXPECT_TRUE(ok);
  double norm = std::sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);
  EXPECT_GT(norm, 1e-6);
}

TEST(PerFootResidualTest, DegeneratePlane) {
  double hip[3] = {0.0, 0.0, 0.0};
  double knee[3] = {0.0, 0.0, 0.0};
  double foot[3] = {0.0, 0.0, 0.0};
  double normal[3] = {0.0, 0.0, 1.0};
  double force[3] = {0.0, 0.0, 80.0};
  double out[3] = {0.0, 0.0, 0.0};
  bool ok = QuadrupedFlatMod::ResidualFn::ComputePerFootResidualForTest(
      force, normal, hip, knee, foot, 0.05, out);
  EXPECT_FALSE(ok);
}
