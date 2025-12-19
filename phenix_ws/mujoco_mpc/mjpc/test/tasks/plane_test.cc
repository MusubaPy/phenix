#include "mjpc/tasks/quadruped_mod/quadruped_mod.h"
#include <gtest/gtest.h>

using mjpc::QuadrupedFlatMod;

TEST(PlaneTest, NonCollinearPoints) {
  double hip[3] = {0.0, 0.0, 0.0};
  double knee[3] = {1.0, 0.0, 0.0};
  double foot[3] = {1.0, 0.0, 1.0};
  double plane[3] = {0.0, 0.0, 0.0};
  bool ok = QuadrupedFlatMod::ResidualFn::ComputePlaneFromPointsForTest(
      hip, knee, foot, plane);
  EXPECT_TRUE(ok);
  // plane normal should be orthogonal to (hip-knee) and (foot-knee)
  double v1[3] = {hip[0]-knee[0], hip[1]-knee[1], hip[2]-knee[2]};
  double v2[3] = {foot[0]-knee[0], foot[1]-knee[1], foot[2]-knee[2]};
  double dot1 = plane[0]*v1[0] + plane[1]*v1[1] + plane[2]*v1[2];
  double dot2 = plane[0]*v2[0] + plane[1]*v2[1] + plane[2]*v2[2];
  EXPECT_NEAR(dot1, 0.0, 1e-6);
  EXPECT_NEAR(dot2, 0.0, 1e-6);
}

TEST(PlaneTest, CollinearPoints) {
  double hip[3] = {0.0, 0.0, 0.0};
  double knee[3] = {1.0, 0.0, 0.0};
  double foot[3] = {2.0, 0.0, 0.0};
  double plane[3] = {0.0, 0.0, 0.0};
  bool ok = QuadrupedFlatMod::ResidualFn::ComputePlaneFromPointsForTest(
      hip, knee, foot, plane);
  EXPECT_FALSE(ok);
}
