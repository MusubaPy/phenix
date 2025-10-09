// Copyright 2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/agent.h"
#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

Task* g_active_task = nullptr;

void ResidualSensorCallback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC && g_active_task != nullptr) {
    g_active_task->Residual(model, data, data->sensordata);
  }
}

struct StabilityMetrics {
  double min_height = std::numeric_limits<double>::infinity();
  double max_abs_roll = 0.0;
  double max_abs_pitch = 0.0;
  double max_planar_speed = 0.0;
};

StabilityMetrics RunQuadrupedScenario(
    const std::function<void(QuadrupedFlat*, mjModel*, mjData*)>& configure,
    double duration_s, int plan_interval) {
  StabilityMetrics metrics;

  Agent agent;
  agent.SetTaskList(GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName("Quadruped Flat");
  if (agent.gui_task_id == -1) {
    ADD_FAILURE() << "Quadruped Flat task not found";
    return metrics;
  }

  Agent::LoadModelResult load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) {
    ADD_FAILURE() << "Failed to load Quadruped Flat model: " << load.error;
    return metrics;
  }
  mjData* data = mj_makeData(model);
  if (data == nullptr) {
    ADD_FAILURE() << "Failed to allocate mjData for Quadruped Flat";
    mjcb_sensor = nullptr;
    g_active_task = nullptr;
    return metrics;
  }

  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) {
    mj_resetDataKeyframe(model, data, home_id);
  }
  mj_forward(model, data);

  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;
  agent.estimator_enabled = false;

  auto* quadruped = dynamic_cast<QuadrupedFlat*>(agent.ActiveTask());
  if (quadruped == nullptr) {
    ADD_FAILURE() << "Active task is not QuadrupedFlat";
    mj_deleteData(data);
    mjcb_sensor = nullptr;
    g_active_task = nullptr;
    return metrics;
  }

  if (configure) {
    configure(quadruped, model, data);
    quadruped->UpdateResidual();
  }

  g_active_task = agent.ActiveTask();
  mjcb_sensor = &ResidualSensorCallback;

  ThreadPool pool(/*num_threads=*/1);

  // Warm-up planning
  agent.ActiveTask()->Transition(model, data);
  agent.state.Set(model, data);
  agent.PlanIteration(&pool);

  const int torso_body_id = mj_name2id(model, mjOBJ_XBODY, "trunk");
  if (torso_body_id < 0) {
    ADD_FAILURE() << "Body 'trunk' not found in model";
    mj_deleteData(data);
    mjcb_sensor = nullptr;
    g_active_task = nullptr;
    return metrics;
  }

  const double timestep = model->opt.timestep;
  const int total_steps = std::max(1, static_cast<int>(duration_s / timestep));
  for (int step = 0; step < total_steps; ++step) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);

    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time(),
        /*use_previous=*/false);

    mj_step(model, data);

    const double* torso_quat = data->xquat + 4 * torso_body_id;
    const double w = torso_quat[0];
    const double x = torso_quat[1];
    const double y = torso_quat[2];
    const double z = torso_quat[3];

    const double sinr_cosp = 2.0 * (w * x + y * z);
    const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    const double roll = std::atan2(sinr_cosp, cosr_cosp);

    const double sinp = 2.0 * (w * y - z * x);
    double pitch = 0.0;
    if (std::abs(sinp) >= 1.0) {
      pitch = std::copysign(mjPI / 2.0, sinp);
    } else {
      pitch = std::asin(sinp);
    }

    metrics.max_abs_roll = std::max(metrics.max_abs_roll, std::abs(roll));
    metrics.max_abs_pitch =
        std::max(metrics.max_abs_pitch, std::abs(pitch));

    const double* torso_pos = data->xipos + 3 * torso_body_id;
    metrics.min_height = std::min(metrics.min_height, torso_pos[2]);

    const double* comvel = SensorByName(model, data, "torso_subtreelinvel");
    if (comvel) {
      const double planar_speed = std::hypot(comvel[0], comvel[1]);
      metrics.max_planar_speed = std::max(metrics.max_planar_speed,
                                          planar_speed);
    }

    if (plan_interval > 0 && step % plan_interval == 0) {
      agent.PlanIteration(&pool);
    }
  }

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  g_active_task = nullptr;

  return metrics;
}

void SetAutomaticGait(QuadrupedFlat* quadruped, const mjModel* model,
                      const mjData*) {
  const int gait_switch_idx =
      ParameterIndex(model, "select_Gait switch");
  ASSERT_GE(gait_switch_idx, 0);
  const int gait_idx = ParameterIndex(model, "select_Gait");
  ASSERT_GE(gait_idx, 0);
  quadruped->parameters[gait_switch_idx] = ReinterpretAsDouble(1);
  quadruped->parameters[gait_idx] = ReinterpretAsDouble(0);
}

void SetManualTrot(QuadrupedFlat* quadruped, const mjModel* model,
                   const mjData*) {
  const int gait_switch_idx =
      ParameterIndex(model, "select_Gait switch");
  ASSERT_GE(gait_switch_idx, 0);
  const int gait_idx = ParameterIndex(model, "select_Gait");
  ASSERT_GE(gait_idx, 0);
  quadruped->parameters[gait_switch_idx] = ReinterpretAsDouble(0);
  quadruped->parameters[gait_idx] = ReinterpretAsDouble(2);
}

void ConfigureAutomaticDebug(QuadrupedFlat* quadruped, const mjModel* model,
                             const mjData*, double walk_speed,
                             double grf_weight, double hind_weight,
                             bool enable_debug) {
  const int gait_switch_idx =
      ParameterIndex(model, "select_Gait switch");
  if (gait_switch_idx >= 0) {
    quadruped->parameters[gait_switch_idx] = ReinterpretAsDouble(1);
  }
  const int gait_idx = ParameterIndex(model, "select_Gait");
  if (gait_idx >= 0) {
    quadruped->parameters[gait_idx] = ReinterpretAsDouble(0);
  }
  const int walk_speed_idx = ParameterIndex(model, "Walk speed");
  if (walk_speed_idx >= 0) {
    quadruped->parameters[walk_speed_idx] = walk_speed;
  }
  const int debug_idx = ParameterIndex(model, "Debug GRF log");
  if (debug_idx >= 0) {
    quadruped->parameters[debug_idx] = enable_debug ? 1.0 : 0.0;
  }
  int grf_idx = CostTermByName(model, "GRF");
  if (grf_idx >= 0 && grf_idx < static_cast<int>(quadruped->weight.size())) {
    quadruped->weight[grf_idx] = grf_weight;
  }
  int hind_idx = CostTermByName(model, "Hind GRF Align");
  if (hind_idx >= 0 && hind_idx < static_cast<int>(quadruped->weight.size())) {
    quadruped->weight[hind_idx] = hind_weight;
  }
}

TEST(QuadrupedStabilityTest, AutomaticGaitMaintainsPose) {
  StabilityMetrics metrics =
      RunQuadrupedScenario(SetAutomaticGait, /*duration_s=*/2.0,
                           /*plan_interval=*/25);
  EXPECT_TRUE(std::isfinite(metrics.min_height));
  EXPECT_GT(metrics.min_height, 0.16);
  EXPECT_LT(metrics.max_abs_pitch, 0.85);
  EXPECT_LT(metrics.max_abs_roll, 0.85);
}

TEST(QuadrupedStabilityTest, ManualTrotRemainsBounded) {
  StabilityMetrics metrics =
      RunQuadrupedScenario(SetManualTrot, /*duration_s=*/1.5,
                           /*plan_interval=*/25);
  EXPECT_TRUE(std::isfinite(metrics.min_height));
  EXPECT_GT(metrics.min_height, 0.1);
  EXPECT_LT(metrics.max_abs_pitch, 1.2);
  EXPECT_LT(metrics.max_abs_roll, 1.2);
}

TEST(QuadrupedStabilityTest, AutomaticGaitDebugWeights) {
  const std::vector<double> weights = {0.005, 0.02};
  for (double w : weights) {
    StabilityMetrics metrics = RunQuadrupedScenario(
        [w](QuadrupedFlat* quadruped, mjModel* model, mjData* data) {
          ConfigureAutomaticDebug(quadruped, model, data,
                                  /*walk_speed=*/0.3, w, w,
                                  /*enable_debug=*/true);
        },
        /*duration_s=*/2.0, /*plan_interval=*/25);
    std::cout << "[SUMMARY] weight=" << w
              << " min_z=" << metrics.min_height
              << " max_abs_roll=" << metrics.max_abs_roll
              << " max_abs_pitch=" << metrics.max_abs_pitch
              << " max_planar_speed=" << metrics.max_planar_speed << std::endl;
  }
  SUCCEED();
}

}  // namespace
}  // namespace mjpc
