// Copyright 2022 DeepMind Technologies Limited
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

#include "mjpc/tasks/quadruped/quadruped.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace {
constexpr double kForceThreshold = 1e-6;
constexpr double kPlaneProjectionEps = 1e-8;
constexpr int kDebugWidth = 10;
constexpr double kDebugPrintInterval = 0.1;   // seconds between debug rows
constexpr int kDebugHeaderRepeat = 20;        // reprint header every N rows
constexpr double kDebugTimeEpsilon = 1e-6;    // tolerance for time comparisons
constexpr double kDebugResetThreshold =
  4 * kDebugPrintInterval;  // difference treated as a real reset

struct FootContactInfo {
  double force[3] = {0.0, 0.0, 0.0};
  double normal[3] = {0.0, 0.0, 0.0};
  double point[3] = {0.0, 0.0, 0.0};
  double weight = 0.0;
  bool in_contact = false;
};

constexpr const char* kFootLabels[4] = {"FL", "HL", "FR", "HR"};

std::mutex& DebugMutex() {
  static std::mutex mutex;
  return mutex;
}

std::string BuildDebugHeader() {
  std::ostringstream oss;
  std::vector<std::string> headers = {
    "time",
    "net_force[x y z]",
    "FL_force[x y z]",
    "HL_force[x y z]",
    "FR_force[x y z]",
    "HR_force[x y z]",
    "align[HL HR]"
  };
  // Индивидуальные ширины столбцов (можно подправить под нужный формат)
  std::vector<int> col_widths = {10, 25, 25, 25, 25, 25, 17};

  auto print_separator = [&]() {
    oss << '+';
    for (size_t i = 0; i < headers.size(); ++i) {
      oss << std::string(col_widths[i], '-') << '+';
    }
    oss << '\n';
  };

  print_separator();
  oss << '|';
  for (size_t i = 0; i < headers.size(); ++i) {
    oss << std::setw(col_widths[i]) << std::left << headers[i] << '|';
  }
  oss << '\n';
  print_separator();
  return oss.str();
}

std::string BuildDebugRow(double time, const double net_force[3],
              const FootContactInfo* contact_info,
              const double hind_alignment[2]) {
  std::ostringstream oss;
  oss << '|'
    << std::setw(kDebugWidth) << std::right << std::fixed
    << std::setprecision(3) << time << '|';

  // net_force as vector
  oss << '['
    << std::setw(kDebugWidth-3) << net_force[0] << ' '
    << std::setw(kDebugWidth-3) << net_force[1] << ' '
    << std::setw(kDebugWidth-3) << net_force[2] << ']'
    << '|';

  // Each foot force as vector
  for (size_t idx = 0; idx < 4; ++idx) {
  const FootContactInfo& info = contact_info[idx];
  oss << '['
    << std::setw(kDebugWidth-3) << info.force[0] << ' '
    << std::setw(kDebugWidth-3) << info.force[1] << ' '
    << std::setw(kDebugWidth-3) << info.force[2] << ']'
    << '|';
  }

  // Hind alignment as vector
  oss << '['
    << std::setw(kDebugWidth-3) << hind_alignment[0] << ' '
    << std::setw(kDebugWidth-3) << hind_alignment[1] << ']'
    << '|';

  oss << '\n';
  return oss.str();
}

void ProjectOntoPlane(double out[3], const double v[3],
                      const double normal[3]) {
  double n[3] = {normal[0], normal[1], normal[2]};
  double norm = mju_norm3(n);
  if (norm < kPlaneProjectionEps) {
    out[0] = out[1] = out[2] = 0.0;
    return;
  }
  mju_scl3(n, n, 1.0 / norm);
  double projection = mju_dot3(v, n);
  out[0] = v[0] - projection * n[0];
  out[1] = v[1] - projection * n[1];
  out[2] = v[2] - projection * n[2];
}

double AngleBetween(const double a[3], const double b[3]) {
  double norm_a = mju_norm3(a);
  double norm_b = mju_norm3(b);
  if (norm_a < kPlaneProjectionEps || norm_b < kPlaneProjectionEps) {
    return 0.0;
  }
  double dot = mju_dot3(a, b) / (norm_a * norm_b);
  dot = mju_clip(dot, -1.0, 1.0);
  return std::acos(dot);
}
}  // namespace

namespace mjpc {
std::string QuadrupedHill::XmlPath() const {
  return GetModelPath("quadruped/task_hill.xml");
}
std::string QuadrupedFlat::XmlPath() const {
  return GetModelPath("quadruped/task_flat.xml");
}
std::string QuadrupedHill::Name() const { return "Quadruped Hill"; }
std::string QuadrupedFlat::Name() const { return "Quadruped Flat"; }

void QuadrupedFlat::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  // start counter
  int counter = 0;

  // get foot positions
  double* foot_pos[kNumFoot];
  for (A1Foot foot : kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

  // average foot position
  double avg_foot_pos[3];
  AverageFootPos(avg_foot_pos, foot_pos);

  FootContactInfo contact_info[kNumFoot];
  double net_grf[3] = {0.0, 0.0, 0.0};

  double* torso_xmat = data->xmat + 9*torso_body_id_;
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  double* compos = SensorByName(model, data, "torso_subtreecom");


  // ---------- Upright ----------
  if (current_mode_ != kModeFlip) {
    if (current_mode_ == kModeBiped) {
      double biped_type = parameters_[biped_type_param_id_];
      int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
      residual[counter++] = torso_xmat[6] - handstand;
    } else {
      residual[counter++] = torso_xmat[8] - 1;
    }
    residual[counter++] = 0;
    residual[counter++] = 0;
  } else {
    // special handling of flip orientation
    double flip_time = data->time - mode_start_time_;
    double quat[4];
    FlipQuat(quat, flip_time);
    double* torso_xquat = data->xquat + 4*torso_body_id_;
    mju_subQuat(residual + counter, torso_xquat, quat);
    counter += 3;
  }


  // ---------- Height ----------
  // quadrupedal or bipedal height of torso over feet
  double* torso_pos = data->xipos + 3*torso_body_id_;
  bool is_biped = current_mode_ == kModeBiped;
  double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
  if (current_mode_ == kModeScramble) {
    // disable height term in Scramble
    residual[counter++] = 0;
  } else if (current_mode_ == kModeFlip) {
    // height target for Backflip
    double flip_time = data->time - mode_start_time_;
    residual[counter++] = torso_pos[2] - FlipHeight(flip_time);
  } else {
    residual[counter++] = (torso_pos[2] - avg_foot_pos[2]) - height_goal;
  }


  // ---------- Position ----------
  double* head = data->site_xpos + 3*head_site_id_;
  double target[3];
  if (current_mode_ == kModeWalk) {
    // follow prescribed Walk trajectory
    double mode_time = data->time - mode_start_time_;
    Walk(target, mode_time);
  } else {
    // go to the goal mocap body
    target[0] = goal_pos[0];
    target[1] = goal_pos[1];
    target[2] = goal_pos[2];
  }
  residual[counter++] = head[0] - target[0];
  residual[counter++] = head[1] - target[1];
  residual[counter++] =
      current_mode_ == kModeScramble ? 2 * (head[2] - target[2]) : 0;

  // ---------- Gait ----------
  A1Gait gait = GetGait();
  double step[kNumFoot];
  FootStep(step, GetPhase(data->time), gait);
  for (A1Foot foot : kFootAll) {
    if (is_biped) {
      // ignore "hands" in biped mode
      bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
      bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
      bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);
      if (front_hand || back_hand) {
        residual[counter++] = 0;
        continue;
      }
    }
    double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};

    if (current_mode_ == kModeScramble) {
      double torso_to_goal[3];
      double* goal = data->mocap_pos + 3*goal_mocap_id_;
      mju_sub3(torso_to_goal, goal, torso_pos);
      mju_normalize3(torso_to_goal);
      mju_sub3(torso_to_goal, goal, foot_pos[foot]);
      torso_to_goal[2] = 0;
      mju_normalize3(torso_to_goal);
      mju_addToScl3(query, torso_to_goal, 0.15);
    }

    double ground_height = Ground(model, data, query);
    double height_target = ground_height + kFootRadius + step[foot];
    double height_difference = foot_pos[foot][2] - height_target;
    if (current_mode_ == kModeScramble) {
      // in Scramble, foot higher than target is not penalized
      height_difference = mju_min(0, height_difference);
    }
    residual[counter++] = step[foot] ? height_difference : 0;
  }


  // ---------- Balance ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double capture_point[3];
  double fall_time = mju_sqrt(2*height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];

  // ---------- Ground reaction force collection ----------
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact& contact = data->contact[i];
    int foot_index = -1;
    double sign = 1.0;
    for (A1Foot candidate : kFootAll) {
      if (foot_geom_id_[candidate] == contact.geom1) {
        foot_index = static_cast<int>(candidate);
        break;
      }
    }
    if (foot_index < 0) {
      for (A1Foot candidate : kFootAll) {
        if (foot_geom_id_[candidate] == contact.geom2) {
          foot_index = static_cast<int>(candidate);
          sign = -1.0;
          break;
        }
      }
    }
    if (foot_index < 0) {
      continue;
    }

    mjtNum contact_force[6];
    mj_contactForce(model, data, i, contact_force);
    mjtNum local_force[3] = {contact_force[0], contact_force[1],
                             contact_force[2]};
    mjtNum world_force_tmp[3];
    mju_mulMatVec(world_force_tmp, contact.frame, local_force, 3, 3);
    double world_force[3] = {static_cast<double>(world_force_tmp[0]),
                             static_cast<double>(world_force_tmp[1]),
                             static_cast<double>(world_force_tmp[2])};
    if (sign < 0) {
      world_force[0] *= -1.0;
      world_force[1] *= -1.0;
      world_force[2] *= -1.0;
    }

    FootContactInfo& info = contact_info[foot_index];
    mju_addTo3(info.force, world_force);
    double force_magnitude = mju_norm3(world_force);
    if (force_magnitude > kForceThreshold) {
      info.in_contact = true;
    }
    info.weight += force_magnitude;
    if (force_magnitude > 0.0) {
      double contact_pos[3] = {
          static_cast<double>(contact.pos[0]),
          static_cast<double>(contact.pos[1]),
          static_cast<double>(contact.pos[2])};
      mju_addToScl3(info.point, contact_pos, force_magnitude);
    }

    double normal_world[3] = {static_cast<double>(contact.frame[6]),
                              static_cast<double>(contact.frame[7]),
                              static_cast<double>(contact.frame[8])};
    if (sign < 0) {
      normal_world[0] *= -1.0;
      normal_world[1] *= -1.0;
      normal_world[2] *= -1.0;
    }
    mju_addTo3(info.normal, normal_world);
    mju_addTo3(net_grf, world_force);
  }

  for (A1Foot foot : kFootAll) {
    FootContactInfo& info = contact_info[foot];
    if (info.weight > kForceThreshold) {
      double inv = 1.0 / info.weight;
      mju_scl3(info.point, info.point, inv);
    } else {
      info.point[0] = foot_pos[foot][0];
      info.point[1] = foot_pos[foot][1];
      info.point[2] = foot_pos[foot][2];
    }
    if (mju_norm3(info.force) < kForceThreshold) {
      info.in_contact = false;
    }
    double norm = mju_norm3(info.normal);
    if (norm > kPlaneProjectionEps) {
      mju_scl3(info.normal, info.normal, 1.0 / norm);
    } else {
      info.normal[0] = info.normal[1] = info.normal[2] = 0.0;
    }
  }


  // ---------- Effort ----------
  mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
  counter += model->nu;


  // ---------- Posture ----------
  double* home = KeyQPosByName(model, data, "home");
  mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
  if (current_mode_ == kModeFlip) {
    double flip_time = data->time - mode_start_time_;
    if (flip_time < crouch_time_) {
      double* crouch = KeyQPosByName(model, data, "crouch");
      mju_sub(residual + counter, data->qpos + 7, crouch + 7, model->nu);
    } else if (flip_time >= crouch_time_ &&
               flip_time < jump_time_ + flight_time_) {
      // free legs during flight phase
      mju_zero(residual + counter, model->nu);
    }
  }
  for (A1Foot foot : kFootAll) {
    for (int joint = 0; joint < 3; joint++) {
      residual[counter + 3*foot + joint] *= kJointPostureGain[joint];
    }
  }
  if (current_mode_ == kModeBiped) {
    // loosen the "hands" in Biped mode
    bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
    double arm_posture = parameters_[arm_posture_param_id_];
    if (handstand) {
      residual[counter + 6] *= arm_posture;
      residual[counter + 7] *= arm_posture;
      residual[counter + 8] *= arm_posture;
      residual[counter + 9] *= arm_posture;
      residual[counter + 10] *= arm_posture;
      residual[counter + 11] *= arm_posture;
    } else {
      residual[counter + 0] *= arm_posture;
      residual[counter + 1] *= arm_posture;
      residual[counter + 2] *= arm_posture;
      residual[counter + 3] *= arm_posture;
      residual[counter + 4] *= arm_posture;
      residual[counter + 5] *= arm_posture;
    }
  }
  counter += model->nu;


  // ---------- Yaw ----------
  double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
  if (current_mode_ == kModeBiped) {
    int handstand =
        ReinterpretAsInt(parameters_[biped_type_param_id_]) ? 1 : -1;
    torso_heading[0] = handstand * torso_xmat[2];
    torso_heading[1] = handstand * torso_xmat[5];
  }
  mju_normalize(torso_heading, 2);
  double heading_goal = parameters_[ParameterIndex(model, "Heading")];
  residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
  residual[counter++] = torso_heading[1] - mju_sin(heading_goal);


  // ---------- Angular momentum ----------
  mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));
  counter +=3;

  // ---------- Net ground reaction force ----------
  double expected_contact[3] = {model->opt.gravity[0], model->opt.gravity[1],
                                model->opt.gravity[2]};
  mju_scl3(expected_contact, expected_contact, -total_mass_);
  double net_residual[3];
  mju_sub3(net_residual, net_grf, expected_contact);
  mju_copy(residual + counter, net_residual, 3);
  counter += 3;

  // ---------- Hind leg GRF alignment ----------
  double hind_alignment[2] = {0.0, 0.0};
  for (int hind_idx = 0; hind_idx < 2; ++hind_idx) {
    A1Foot foot = kFootHind[hind_idx];
    FootContactInfo& info = contact_info[foot];
    if (!info.in_contact) {
      // Only the hind-left foot contributes to the alignment cost.
      hind_alignment[hind_idx] = 0.0;
      continue;
    }

    int abduction_joint = abduction_joint_id_[foot];
    int hip_joint = hip_joint_id_[foot];
    int knee_joint = knee_joint_id_[foot];
    if (abduction_joint < 0 || hip_joint < 0 || knee_joint < 0) {
      hind_alignment[hind_idx] = 0.0;
      continue;
    }

  // `xanchor` is stored per joint (njnt), so index by joint id directly.
  const mjtNum* abduction_anchor_ptr =
    data->xanchor + 3 * abduction_joint;
  const mjtNum* hip_anchor_ptr =
    data->xanchor + 3 * hip_joint;
  const mjtNum* knee_anchor_ptr =
    data->xanchor + 3 * knee_joint;
    double anchors[3][3];
    mju_copy3(anchors[0], abduction_anchor_ptr);
    mju_copy3(anchors[1], hip_anchor_ptr);
    mju_copy3(anchors[2], knee_anchor_ptr);

    double foot_point[3];
    if (info.in_contact) {
      mju_copy3(foot_point, info.point);
    } else {
      mju_copy3(foot_point, foot_pos[foot]);
    }

    double plane_vec1[3];  // hip -> knee
    double plane_vec2[3];  // knee -> foot contact
    mju_sub3(plane_vec1, anchors[1], anchors[2]);
    mju_sub3(plane_vec2, foot_point, anchors[2]);
    double plane_normal[3];
    mju_cross(plane_normal, plane_vec1, plane_vec2);
    if (mju_norm3(plane_normal) < kPlaneProjectionEps) {
      hind_alignment[hind_idx] = 0.0;
      continue;
    }

    if (mju_dot3(plane_normal, info.normal) < 0) {
      mju_scl3(plane_normal, plane_normal, -1.0);
    }

    double normal_proj[3];
    ProjectOntoPlane(normal_proj, info.normal, plane_normal);
    if (mju_norm3(normal_proj) < kPlaneProjectionEps) {
      hind_alignment[hind_idx] = 0.0;
      continue;
    }

    double best_motor_proj[3] = {0.0, 0.0, 0.0};
    double smallest_angle = mjMAXVAL;
    double candidate_vectors[2][3];
    mju_sub3(candidate_vectors[0], anchors[2], anchors[1]);       // hip -> knee
    mju_sub3(candidate_vectors[1], foot_point, anchors[2]);       // knee -> foot
    for (int candidate = 0; candidate < 2; ++candidate) {
      double motor_proj[3];
      ProjectOntoPlane(motor_proj, candidate_vectors[candidate], plane_normal);
      if (mju_norm3(motor_proj) < kPlaneProjectionEps) {
        continue;
      }
      double angle = AngleBetween(normal_proj, motor_proj);
      if (angle < smallest_angle) {
        smallest_angle = angle;
        mju_copy3(best_motor_proj, motor_proj);
      }
    }
    if (smallest_angle == mjMAXVAL) {
      hind_alignment[hind_idx] = 0.0;
      continue;
    }

  double grf_proj[3];
  ProjectOntoPlane(grf_proj, info.force, plane_normal);
    if (mju_norm3(grf_proj) < kPlaneProjectionEps) {
      hind_alignment[hind_idx] = 0.0;
      continue;
    }

    hind_alignment[hind_idx] = AngleBetween(grf_proj, best_motor_proj);
  }

  bool debug_enabled =
      debug_grf_param_id_ >= 0 && parameters_[debug_grf_param_id_] > 0.5;
  if (debug_enabled) {
    double current_time = data->time;
    bool should_print = false;
    bool print_header = false;

    auto state = debug_log_state_;
    {
      std::lock_guard<std::mutex> state_lock(state->state_mutex);

      if (!std::isfinite(state->next_print_time)) {
        state->next_print_time = current_time;
      }

      double time_delta = state->last_print_time - current_time;
      bool out_of_order_sample = false;

      if (time_delta > kDebugTimeEpsilon) {
        if (time_delta > kDebugResetThreshold) {
          state->header_printed = false;
          state->print_count = 0;
          state->next_print_time = current_time;
          state->last_print_time = current_time;
        } else {
          out_of_order_sample = true;
        }
      } else {
        state->last_print_time = current_time;
      }

      if (!out_of_order_sample &&
          current_time + kDebugTimeEpsilon >= state->next_print_time) {
        should_print = true;
        print_header = !state->header_printed ||
                       (state->print_count % kDebugHeaderRepeat) == 0;
        state->header_printed = true;
        state->print_count++;
        state->last_print_time = current_time;
        state->next_print_time = current_time + kDebugPrintInterval;
      }
    }

    if (should_print) {
      std::lock_guard<std::mutex> lock(DebugMutex());
      if (print_header) {
        std::cout << BuildDebugHeader();
      }
      std::cout << BuildDebugRow(current_time, net_grf, contact_info,
                                 hind_alignment)
                << std::flush;
    }
  } else {
    auto state = debug_log_state_;
    std::lock_guard<std::mutex> state_lock(state->state_mutex);
    state->header_printed = false;
    state->print_count = 0;
    state->last_print_time = -std::numeric_limits<double>::infinity();
    state->next_print_time = -std::numeric_limits<double>::infinity();
  }

  if (hind_grf_align_sensor_id_ >= 0) {
    residual[counter++] = hind_alignment[0];
    residual[counter++] = hind_alignment[1];
  }

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

//  ============  transition  ============
void QuadrupedFlat::TransitionLocked(mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
  if (data->time < residual_.last_transition_time_ ||
      residual_.last_transition_time_ == -1) {
    if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
      mode = ResidualFn::kModeQuadruped;  // mode stateful, switch to Quadruped
    }
    residual_.last_transition_time_ = residual_.phase_start_time_ =
        residual_.phase_start_ = data->time;
  }

  // ---------- prevent forbidden mode transitions ----------
  // switching mode, not from quadruped
  if (mode != residual_.current_mode_ &&
      residual_.current_mode_ != ResidualFn::kModeQuadruped) {
    // switch into stateful mode only allowed from Quadruped
    if (mode == ResidualFn::kModeWalk || mode == ResidualFn::kModeFlip) {
      mode = ResidualFn::kModeQuadruped;
    }
  }

  // ---------- handle phase velocity change ----------
  double phase_velocity = 2 * mjPI * parameters[residual_.cadence_param_id_];
  if (phase_velocity != residual_.phase_velocity_) {
    residual_.phase_start_ = residual_.GetPhase(data->time);
    residual_.phase_start_time_ = data->time;
    residual_.phase_velocity_ = phase_velocity;
  }


  // ---------- automatic gait switching ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double beta = mju_exp(-(data->time - residual_.last_transition_time_) /
                        ResidualFn::kAutoGaitFilter);
  residual_.com_vel_[0] = beta * residual_.com_vel_[0] + (1 - beta) * comvel[0];
  residual_.com_vel_[1] = beta * residual_.com_vel_[1] + (1 - beta) * comvel[1];
  // TODO(b/268398978): remove reinterpret, int64_t business
  int auto_switch =
      ReinterpretAsInt(parameters[residual_.gait_switch_param_id_]);
  if (mode == ResidualFn::kModeBiped) {
    // biped always trots
    parameters[residual_.gait_param_id_] =
        ReinterpretAsDouble(ResidualFn::kGaitTrot);
  } else if (auto_switch) {
    double com_speed = mju_norm(residual_.com_vel_, 2);
    for (int64_t gait : ResidualFn::kGaitAll) {
      // scramble requires a non-static gait
      if (mode == ResidualFn::kModeScramble && gait == ResidualFn::kGaitStand)
        continue;
      bool lower = com_speed > ResidualFn::kGaitAuto[gait];
      bool upper = gait == ResidualFn::kGaitGallop ||
                   com_speed <= ResidualFn::kGaitAuto[gait + 1];
      bool wait = mju_abs(residual_.gait_switch_time_ - data->time) >
                  ResidualFn::kAutoGaitMinTime;
      if (lower && upper && wait) {
        parameters[residual_.gait_param_id_] = ReinterpretAsDouble(gait);
        residual_.gait_switch_time_ = data->time;
      }
    }
  }


  // ---------- handle gait switch, manual or auto ----------
  double gait_selection = parameters[residual_.gait_param_id_];
  if (gait_selection != residual_.current_gait_) {
    residual_.current_gait_ = gait_selection;
    ResidualFn::A1Gait gait = residual_.GetGait();
    parameters[residual_.duty_param_id_] = ResidualFn::kGaitParam[gait][0];
    parameters[residual_.cadence_param_id_] = ResidualFn::kGaitParam[gait][1];
    parameters[residual_.amplitude_param_id_] = ResidualFn::kGaitParam[gait][2];
    weight[residual_.balance_cost_id_] = ResidualFn::kGaitParam[gait][3];
    weight[residual_.upright_cost_id_] = ResidualFn::kGaitParam[gait][4];
    weight[residual_.height_cost_id_] = ResidualFn::kGaitParam[gait][5];
  }


  // ---------- Walk ----------
  double* goal_pos = data->mocap_pos + 3*residual_.goal_mocap_id_;
  if (mode == ResidualFn::kModeWalk) {
    double angvel = parameters[ParameterIndex(model, "Walk turn")];
    double speed = parameters[ParameterIndex(model, "Walk speed")];

    // current torso direction
    double* torso_xmat = data->xmat + 9*residual_.torso_body_id_;
    double forward[2] = {torso_xmat[0], torso_xmat[3]};
    mju_normalize(forward, 2);
    double leftward[2] = {-forward[1], forward[0]};

    // switching into Walk or parameters changed, reset task state
    if (mode != residual_.current_mode_ || residual_.angvel_ != angvel ||
        residual_.speed_ != speed) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save current speed and angvel
      residual_.speed_ = speed;
      residual_.angvel_ = angvel;

      // compute and save rotation axis / walk origin
      double axis[2] = {data->xpos[3*residual_.torso_body_id_],
                        data->xpos[3*residual_.torso_body_id_+1]};
      if (mju_abs(angvel) > ResidualFn::kMinAngvel) {
        // don't allow turning with very small angvel
        double d = speed / angvel;
        axis[0] += d * leftward[0];
        axis[1] += d * leftward[1];
      }
      residual_.position_[0] = axis[0];
      residual_.position_[1] = axis[1];

      // save vector from axis to initial goal position
      residual_.heading_[0] = goal_pos[0] - axis[0];
      residual_.heading_[1] = goal_pos[1] - axis[1];
    }

    // move goal
    double time = data->time - residual_.mode_start_time_;
    residual_.Walk(goal_pos, time);
  }


  // ---------- Flip ----------
  double* compos = SensorByName(model, data, "torso_subtreecom");
  if (mode == ResidualFn::kModeFlip) {
    // switching into Flip, reset task state
    if (mode != residual_.current_mode_) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save body orientation, ground height
      mju_copy4(residual_.orientation_,
                data->xquat + 4 * residual_.torso_body_id_);
      residual_.ground_ = Ground(model, data, compos);

      // save parameters
      residual_.save_weight_ = weight;
      residual_.save_gait_switch_ = parameters[residual_.gait_switch_param_id_];

      // set parameters
      weight[CostTermByName(model, "Upright")] = 0.2;
      weight[CostTermByName(model, "Height")] = 5;
      weight[CostTermByName(model, "Position")] = 0;
      weight[CostTermByName(model, "Gait")] = 0;
      weight[CostTermByName(model, "Balance")] = 0;
      weight[CostTermByName(model, "Effort")] = 0.005;
      weight[CostTermByName(model, "Posture")] = 0.1;
      parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(1);
    }

    // time from start of Flip
    double flip_time = data->time - residual_.mode_start_time_;

    if (flip_time >=
        residual_.jump_time_ + residual_.flight_time_ + residual_.land_time_) {
      // Flip ended, back to Quadruped, restore values
      mode = ResidualFn::kModeQuadruped;
      weight = residual_.save_weight_;
      parameters[residual_.gait_switch_param_id_] = residual_.save_gait_switch_;
      goal_pos[0] = data->site_xpos[3*residual_.head_site_id_ + 0];
      goal_pos[1] = data->site_xpos[3*residual_.head_site_id_ + 1];
    }
  }

  // save mode
  residual_.current_mode_ = static_cast<ResidualFn::A1Mode>(mode);
  residual_.last_transition_time_ = data->time;
}

// colors of visualisation elements drawn in ModifyScene()
constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};  // step-height cylinders
constexpr float kHullRgba[4] = {0.4, 0.2, 0.8, 1};  // convex hull
constexpr float kAvgRgba[4] = {0.4, 0.2, 0.8, 1};   // average foot position
constexpr float kCapRgba[4] = {0.3, 0.3, 0.8, 1};   // capture point
constexpr float kPcpRgba[4] = {0.5, 0.5, 0.2, 1};   // projected capture point
constexpr float kPlaneRgba[4] = {0.2f, 0.8f, 0.8f, 0.25f};            // contact plane
constexpr float kGrfVectorRgba[4] = {0.95f, 0.25f, 0.25f, 1.0f};   // GRF vector
constexpr float kMotorVectorRgba[4] = {0.2f, 0.55f, 0.95f, 1.0f};  // motor vector
constexpr float kPlanePointRgba[3][4] = {
  {0.85f, 0.25f, 0.25f, 1.0f},  // hip anchor
  {0.25f, 0.85f, 0.25f, 1.0f},  // knee anchor
  {0.25f, 0.25f, 0.85f, 1.0f}   // contact / foot point
};
constexpr double kGrfVectorScale = 1; //5e-4;                            // GRF scaling
constexpr double kVectorMaxLength = 0.3;                            // max vector len
constexpr double kVectorWidth = 0.01;                              // vector thickness

// draw task-related geometry in the scene
void QuadrupedFlat::ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {
  // flip target pose
  if (residual_.current_mode_ == ResidualFn::kModeFlip) {
    double flip_time = data->time - residual_.mode_start_time_;
    double* torso_pos = data->xpos + 3*residual_.torso_body_id_;
    double pos[3] = {torso_pos[0], torso_pos[1],
                     residual_.FlipHeight(flip_time)};
    double quat[4];
    residual_.FlipQuat(quat, flip_time);
    double mat[9];
    mju_quat2Mat(mat, quat);
    double size[3] = {0.25, 0.15, 0.05};
    float rgba[4] = {0, 1, 0, 0.5};
    AddGeom(scene, mjGEOM_BOX, size, pos, mat, rgba);

    // don't draw anything else during flip
    return;
  }

  // current foot positions
  double* foot_pos[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * residual_.foot_geom_id_[foot];

  FootContactInfo contact_info[ResidualFn::kNumFoot];

  for (int i = 0; i < data->ncon; ++i) {
    const mjContact& contact = data->contact[i];
    int foot_index = -1;
    double sign = 1.0;

    for (ResidualFn::A1Foot candidate : ResidualFn::kFootAll) {
      if (residual_.foot_geom_id_[candidate] == contact.geom1) {
        foot_index = static_cast<int>(candidate);
        break;
      }
    }
    if (foot_index < 0) {
      for (ResidualFn::A1Foot candidate : ResidualFn::kFootAll) {
        if (residual_.foot_geom_id_[candidate] == contact.geom2) {
          foot_index = static_cast<int>(candidate);
          sign = -1.0;
          break;
        }
      }
    }
    if (foot_index < 0) continue;

    mjtNum contact_force[6];
    mj_contactForce(model, data, i, contact_force);
    mjtNum local_force[3] = {contact_force[0], contact_force[1], contact_force[2]};
    mjtNum world_force_tmp[3];
    mju_mulMatVec(world_force_tmp, contact.frame, local_force, 3, 3);
    double world_force[3] = {static_cast<double>(world_force_tmp[0]),
                             static_cast<double>(world_force_tmp[1]),
                             static_cast<double>(world_force_tmp[2])};
    if (sign < 0) {
      world_force[0] *= -1.0;
      world_force[1] *= -1.0;
      world_force[2] *= -1.0;
    }

    FootContactInfo& info = contact_info[foot_index];
    mju_addTo3(info.force, world_force);
    double force_magnitude = mju_norm3(world_force);
    if (force_magnitude > kForceThreshold) {
      info.in_contact = true;
    }
    info.weight += force_magnitude;
    if (force_magnitude > 0.0) {
      double contact_pos[3] = {static_cast<double>(contact.pos[0]),
                               static_cast<double>(contact.pos[1]),
                               static_cast<double>(contact.pos[2])};
      mju_addToScl3(info.point, contact_pos, force_magnitude);
    }

    double normal_world[3] = {static_cast<double>(contact.frame[6]),
                              static_cast<double>(contact.frame[7]),
                              static_cast<double>(contact.frame[8])};
    if (sign < 0) {
      normal_world[0] *= -1.0;
      normal_world[1] *= -1.0;
      normal_world[2] *= -1.0;
    }
    mju_addTo3(info.normal, normal_world);
  }

  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    FootContactInfo& info = contact_info[foot];
    if (info.weight > kForceThreshold) {
      double inv = 1.0 / info.weight;
      mju_scl3(info.point, info.point, inv);
    } else {
      info.point[0] = foot_pos[foot][0];
      info.point[1] = foot_pos[foot][1];
      info.point[2] = foot_pos[foot][2];
    }
    if (mju_norm3(info.force) < kForceThreshold) {
      info.in_contact = false;
    }
    double norm = mju_norm3(info.normal);
    if (norm > kPlaneProjectionEps) {
      mju_scl3(info.normal, info.normal, 1.0 / norm);
    } else {
      info.normal[0] = info.normal[1] = info.normal[2] = 0.0;
    }
  }

  // stance and flight positions
  double flight_pos[ResidualFn::kNumFoot][3];
  double stance_pos[ResidualFn::kNumFoot][3];
  // set to foot horizontal position:
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    flight_pos[foot][0] = stance_pos[foot][0] = foot_pos[foot][0];
    flight_pos[foot][1] = stance_pos[foot][1] = foot_pos[foot][1];
  }

  // ground height below feet
  double ground[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    ground[foot] = Ground(model, data, foot_pos[foot]);
  }

  // step heights
  ResidualFn::A1Gait gait = residual_.GetGait();
  double step[ResidualFn::kNumFoot];
  residual_.FootStep(step, residual_.GetPhase(data->time), gait);

  // draw step height
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    stance_pos[foot][2] = ResidualFn::kFootRadius + ground[foot];
    if (residual_.current_mode_ == ResidualFn::kModeBiped) {
      // skip "hands" in biped mode
      bool handstand =
          ReinterpretAsInt(parameters[residual_.biped_type_param_id_]);
      bool front_hand = !handstand && (foot == ResidualFn::kFootFL ||
                                       foot == ResidualFn::kFootFR);
      bool back_hand = handstand && (foot == ResidualFn::kFootHL ||
                                     foot == ResidualFn::kFootHR);
      if (front_hand || back_hand) continue;
    }
    if (step[foot]) {
      flight_pos[foot][2] = ResidualFn::kFootRadius + step[foot] + ground[foot];
      AddConnector(scene, mjGEOM_CYLINDER, ResidualFn::kFootRadius,
                   stance_pos[foot], flight_pos[foot], kStepRgba);
    }
  }

  // support polygon (currently unused for cost)
  double polygon[2*ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    polygon[2*foot] = foot_pos[foot][0];
    polygon[2*foot + 1] = foot_pos[foot][1];
  }
  int hull[ResidualFn::kNumFoot];
  int num_hull = Hull2D(hull, ResidualFn::kNumFoot, polygon);
  for (int i=0; i < num_hull; i++) {
    int j = (i + 1) % num_hull;
    AddConnector(scene, mjGEOM_CAPSULE, ResidualFn::kFootRadius/2,
                 stance_pos[hull[i]], stance_pos[hull[j]], kHullRgba);
  }

  // capture point
  bool is_biped = residual_.current_mode_ == ResidualFn::kModeBiped;
  double height_goal =
      is_biped ? ResidualFn::kHeightBiped : ResidualFn::kHeightQuadruped;
  double fall_time = mju_sqrt(2*height_goal / residual_.gravity_);
  double capture[3];
  double* compos = SensorByName(model, data, "torso_subtreecom");
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  mju_addScl3(capture, compos, comvel, fall_time);

  // ground under CoM
  double com_ground = Ground(model, data, compos);

  // average foot position
  double feet_pos[3];
  residual_.AverageFootPos(feet_pos, foot_pos);
  feet_pos[2] = com_ground;

  double foot_size[3] = {ResidualFn::kFootRadius, 0, 0};

  // average foot position
  AddGeom(scene, mjGEOM_SPHERE, foot_size, feet_pos, /*mat=*/nullptr, kAvgRgba);

  // capture point
  capture[2] = com_ground;
  AddGeom(scene, mjGEOM_SPHERE, foot_size, capture, /*mat=*/nullptr, kCapRgba);

  // capture point, projected onto hull
  double pcp2[2];
  NearestInHull(pcp2, capture, polygon, hull, num_hull);
  double pcp[3] = {pcp2[0], pcp2[1], com_ground};
  AddGeom(scene, mjGEOM_SPHERE, foot_size, pcp, /*mat=*/nullptr, kPcpRgba);

  auto visualize_alignment_plane = [&](ResidualFn::A1Foot foot) {
    FootContactInfo& info = contact_info[foot];
    // Visualize the alignment plane only for the hind-left foot.
    int knee_joint = residual_.knee_joint_id_[foot];
    int hip_joint = residual_.hip_joint_id_[foot];
    if (knee_joint < 0 || hip_joint < 0) {
      return;
    }

    // `xanchor` entries are laid out per joint (njnt).
    const mjtNum* knee_anchor_ptr = data->xanchor + 3 * knee_joint;
    const mjtNum* hip_anchor_ptr = data->xanchor + 3 * hip_joint;

    double knee_anchor[3];
    double hip_anchor[3];
    mju_copy3(knee_anchor, knee_anchor_ptr);
    mju_copy3(hip_anchor, hip_anchor_ptr);

    double foot_point[3];
    if (info.in_contact) {
      mju_copy3(foot_point, info.point);
    } else {
      mju_copy3(foot_point, foot_pos[foot]);
    }

    double plane_vec1[3];  // hip -> knee
    double plane_vec2[3];  // knee -> foot
    mju_sub3(plane_vec1, hip_anchor, knee_anchor);
    mju_sub3(plane_vec2, foot_point, knee_anchor);
    double plane_normal[3];
    mju_cross(plane_normal, plane_vec1, plane_vec2);
    if (mju_norm3(plane_normal) < kPlaneProjectionEps) {
      return;
    }
    double vec1_norm = mju_norm3(plane_vec1);
    double vec2_norm = mju_norm3(plane_vec2);
    if (vec1_norm < kPlaneProjectionEps || vec2_norm < kPlaneProjectionEps) {
      return;
    }

    double plane_u[3];
    double plane_v[3];
    mju_copy3(plane_u, plane_vec1);
    mju_copy3(plane_v, plane_vec2);
    mju_scl3(plane_u, plane_u, 1.0 / vec1_norm);
    mju_scl3(plane_v, plane_v, 1.0 / vec2_norm);
    double plane_normal_unit[3];
    mju_copy3(plane_normal_unit, plane_normal);
    mju_normalize3(plane_normal_unit);

    mjtNum plane_center[3] = {
        static_cast<mjtNum>((foot_point[0] + knee_anchor[0] + hip_anchor[0]) /
                            3.0),
        static_cast<mjtNum>((foot_point[1] + knee_anchor[1] + hip_anchor[1]) /
                            3.0),
        static_cast<mjtNum>((foot_point[2] + knee_anchor[2] + hip_anchor[2]) /
                            3.0)};
    mju_addToScl3(plane_center, plane_normal_unit, 0.001);

    double size0 = vec1_norm;
    double size1 = vec2_norm;
    mjtNum plane_size[3] = {
        static_cast<mjtNum>(mju_max(size0, 0.02)),
        static_cast<mjtNum>(mju_max(size1, 0.02)),
        0.001};
    mjtNum plane_mat[9] = {
        static_cast<mjtNum>(plane_u[0]),
        static_cast<mjtNum>(plane_v[0]),
        static_cast<mjtNum>(plane_normal_unit[0]),
        static_cast<mjtNum>(plane_u[1]),
        static_cast<mjtNum>(plane_v[1]),
        static_cast<mjtNum>(plane_normal_unit[1]),
        static_cast<mjtNum>(plane_u[2]),
        static_cast<mjtNum>(plane_v[2]),
        static_cast<mjtNum>(plane_normal_unit[2])};

    AddGeom(scene, mjGEOM_BOX, plane_size, plane_center, plane_mat,
            kPlaneRgba);

    double point_size[3] = {0.03, 0.0, 0.0};
    mjtNum hip_point[3] = {static_cast<mjtNum>(hip_anchor[0]),
                           static_cast<mjtNum>(hip_anchor[1]),
                           static_cast<mjtNum>(hip_anchor[2])};
    mjtNum knee_point[3] = {static_cast<mjtNum>(knee_anchor[0]),
                            static_cast<mjtNum>(knee_anchor[1]),
                            static_cast<mjtNum>(knee_anchor[2])};
    mjtNum contact_point[3] = {static_cast<mjtNum>(foot_point[0]),
                               static_cast<mjtNum>(foot_point[1]),
                               static_cast<mjtNum>(foot_point[2])};
    AddGeom(scene, mjGEOM_SPHERE, point_size, hip_point, /*mat=*/nullptr,
            kPlanePointRgba[0]);
    AddGeom(scene, mjGEOM_SPHERE, point_size, knee_point, /*mat=*/nullptr,
            kPlanePointRgba[1]);
    AddGeom(scene, mjGEOM_SPHERE, point_size, contact_point, /*mat=*/nullptr,
            kPlanePointRgba[2]);

    if (!info.in_contact) {
      return;
    }

    double grf_proj[3];
    ProjectOntoPlane(grf_proj, info.force, plane_normal_unit);
    double grf_proj_norm = mju_norm3(grf_proj);
    if (grf_proj_norm < kPlaneProjectionEps) {
      return;
    }
    double grf_dir[3];
    mju_copy3(grf_dir, grf_proj);
    mju_scl3(grf_dir, grf_dir, 1.0 / grf_proj_norm);

    double thigh_vec[3];
    double shank_vec[3];
    mju_sub3(thigh_vec, knee_anchor, hip_anchor);   // hip -> knee
    mju_sub3(shank_vec, foot_point, knee_anchor);   // knee -> foot

    double thigh_proj[3];
    double shank_proj[3];
    ProjectOntoPlane(thigh_proj, thigh_vec, plane_normal_unit);
    ProjectOntoPlane(shank_proj, shank_vec, plane_normal_unit);

    double best_motor_proj[3] = {0.0, 0.0, 0.0};
    double min_perp_distance = mjMAXVAL;
    bool has_motor = false;

    auto consider_motor = [&](const double motor_proj[3]) {
      double norm = mju_norm3(motor_proj);
      if (norm < kPlaneProjectionEps) return;

      double parallel = mju_dot3(motor_proj, grf_dir);
      if (parallel <= 0.0) {
        // Ранее мы принимали такие моторы и вектор визуализации мог развернуться
        // назад вдоль пола. Игнорируем их, чтобы оставался только мотор,
        // поддерживающий направление проекции ГРФ.
        return;
      }

      double closest_point[3];
      mju_scl3(closest_point, grf_dir, parallel);
      double diff[3];
      mju_sub3(diff, motor_proj, closest_point);
      double distance = mju_norm3(diff);
      if (distance < min_perp_distance) {
        min_perp_distance = distance;
        mju_copy3(best_motor_proj, motor_proj);
        has_motor = true;
      }
    };

    consider_motor(thigh_proj);
    consider_motor(shank_proj);

    if (!has_motor) {
      return;
    }

    double grf_draw[3];
    mju_copy3(grf_draw, grf_proj);
    mju_scl3(grf_draw, grf_draw, kGrfVectorScale);
    double grf_length = mju_norm3(grf_draw);
    if (grf_length > kVectorMaxLength && grf_length > 0) {
      mju_scl3(grf_draw, grf_draw, kVectorMaxLength / grf_length);
    }

    double motor_draw[3];
    mju_copy3(motor_draw, best_motor_proj);
    double motor_length = mju_norm3(motor_draw);
    if (motor_length > kVectorMaxLength && motor_length > 0) {
      mju_scl3(motor_draw, motor_draw, kVectorMaxLength / motor_length);
    }

    mjtNum from[3] = {foot_point[0], foot_point[1], foot_point[2]};
    mjtNum grf_to[3] = {foot_point[0] + grf_draw[0],
                        foot_point[1] + grf_draw[1],
                        foot_point[2] + grf_draw[2]};
    mjtNum motor_from[3] = {static_cast<mjtNum>(foot_point[0]),
                            static_cast<mjtNum>(foot_point[1]),
                            static_cast<mjtNum>(foot_point[2])};
    mjtNum motor_to[3] = {
        static_cast<mjtNum>(foot_point[0] + motor_draw[0]),
        static_cast<mjtNum>(foot_point[1] + motor_draw[1]),
        static_cast<mjtNum>(foot_point[2] + motor_draw[2])};

    AddConnector(scene, mjGEOM_CAPSULE, kVectorWidth, from, grf_to,
                 kGrfVectorRgba);
    AddConnector(scene, mjGEOM_CAPSULE, kVectorWidth, motor_from, motor_to,
                 kMotorVectorRgba);
  };

  for (int hind_idx = 0; hind_idx < 2; ++hind_idx) {
    visualize_alignment_plane(ResidualFn::kFootHind[hind_idx]);
  }
}

//  ============  task-state utilities  ============
// save task-related ids
void QuadrupedFlat::ResetLocked(const mjModel* model) {
  // ----------  task identifiers  ----------
  residual_.gait_param_id_ = ParameterIndex(model, "select_Gait");
  residual_.gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
  residual_.flip_dir_param_id_ = ParameterIndex(model, "select_Flip dir");
  residual_.biped_type_param_id_ = ParameterIndex(model, "select_Biped type");
  residual_.cadence_param_id_ = ParameterIndex(model, "Cadence");
  residual_.amplitude_param_id_ = ParameterIndex(model, "Amplitude");
  residual_.duty_param_id_ = ParameterIndex(model, "Duty ratio");
  residual_.arm_posture_param_id_ = ParameterIndex(model, "Arm posture");
  residual_.balance_cost_id_ = CostTermByName(model, "Balance");
  residual_.upright_cost_id_ = CostTermByName(model, "Upright");
  residual_.height_cost_id_ = CostTermByName(model, "Height");

  // ----------  model identifiers  ----------
  residual_.torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "trunk");
  if (residual_.torso_body_id_ < 0) mju_error("body 'trunk' not found");

  residual_.head_site_id_ = mj_name2id(model, mjOBJ_SITE, "head");
  if (residual_.head_site_id_ < 0) mju_error("site 'head' not found");

  int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_id < 0) mju_error("body 'goal' not found");

  residual_.goal_mocap_id_ = model->body_mocapid[goal_id];
  if (residual_.goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

  // foot geom ids
  int foot_index = 0;
  for (const char* footname : {"FL", "HL", "FR", "HR"}) {
    int foot_id = mj_name2id(model, mjOBJ_GEOM, footname);
    if (foot_id < 0) mju_error_s("geom '%s' not found", footname);
    residual_.foot_geom_id_[foot_index] = foot_id;
    foot_index++;
  }

  // shoulder body ids
  int shoulder_index = 0;
  for (const char* shouldername : {"FL_hip", "HL_hip", "FR_hip", "HR_hip"}) {
    int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
    if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
    residual_.shoulder_body_id_[shoulder_index] = foot_id;
    shoulder_index++;
  }

  residual_.debug_grf_param_id_ = ParameterIndex(model, "Debug GRF log");
  residual_.hind_grf_align_sensor_id_ =
      mj_name2id(model, mjOBJ_SENSOR, "Hind GRF Align");
  {
    auto state = residual_.debug_log_state_;
    std::lock_guard<std::mutex> state_lock(state->state_mutex);
    state->header_printed = false;
    state->print_count = 0;
    state->last_print_time = -std::numeric_limits<double>::infinity();
    state->next_print_time = -std::numeric_limits<double>::infinity();
  }

  for (int foot = 0; foot < ResidualFn::kNumFoot; ++foot) {
    const char* abduction_joint =
        ResidualFn::kAbductionJointNames[foot];
    residual_.abduction_joint_id_[foot] =
        mj_name2id(model, mjOBJ_JOINT, abduction_joint);
    if (residual_.abduction_joint_id_[foot] < 0) {
      mju_error_s("joint '%s' not found", abduction_joint);
    }
    const char* hip_joint = ResidualFn::kHipJointNames[foot];
    residual_.hip_joint_id_[foot] =
        mj_name2id(model, mjOBJ_JOINT, hip_joint);
    if (residual_.hip_joint_id_[foot] < 0) {
      mju_error_s("joint '%s' not found", hip_joint);
    }
    const char* knee_joint = ResidualFn::kKneeJointNames[foot];
    residual_.knee_joint_id_[foot] =
        mj_name2id(model, mjOBJ_JOINT, knee_joint);
    if (residual_.knee_joint_id_[foot] < 0) {
      mju_error_s("joint '%s' not found", knee_joint);
    }
  }

  residual_.total_mass_ = 0.0;
  for (int i = 0; i < model->nbody; ++i) {
    residual_.total_mass_ += model->body_mass[i];
  }

  // ----------  derived kinematic quantities for Flip  ----------
  residual_.gravity_ = mju_norm3(model->opt.gravity);
  // velocity at takeoff
  residual_.jump_vel_ =
      mju_sqrt(2 * residual_.gravity_ *
               (ResidualFn::kMaxHeight - ResidualFn::kLeapHeight));
  // time in flight phase
  residual_.flight_time_ = 2 * residual_.jump_vel_ / residual_.gravity_;
  // acceleration during jump phase
  residual_.jump_acc_ =
      residual_.jump_vel_ * residual_.jump_vel_ /
      (2 * (ResidualFn::kLeapHeight - ResidualFn::kCrouchHeight));
  // time in crouch sub-phase of jump
  residual_.crouch_time_ =
      mju_sqrt(2 * (ResidualFn::kHeightQuadruped - ResidualFn::kCrouchHeight) /
               residual_.jump_acc_);
  // time in leap sub-phase of jump
  residual_.leap_time_ = residual_.jump_vel_ / residual_.jump_acc_;
  // jump total time
  residual_.jump_time_ = residual_.crouch_time_ + residual_.leap_time_;
  // velocity at beginning of crouch
  residual_.crouch_vel_ = -residual_.jump_acc_ * residual_.crouch_time_;
  // time of landing phase
  residual_.land_time_ =
      2 * (ResidualFn::kLeapHeight - ResidualFn::kHeightQuadruped) /
      residual_.jump_vel_;
  // acceleration during landing
  residual_.land_acc_ = residual_.jump_vel_ / residual_.land_time_;
  // rotational velocity during flight phase (rotates 1.25 pi)
  residual_.flight_rot_vel_ = 1.25 * mjPI / residual_.flight_time_;
  // rotational velocity at start of leap (rotates 0.5 pi)
  residual_.jump_rot_vel_ =
      mjPI / residual_.leap_time_ - residual_.flight_rot_vel_;
  // rotational acceleration during leap (rotates 0.5 pi)
  residual_.jump_rot_acc_ =
      (residual_.flight_rot_vel_ - residual_.jump_rot_vel_) /
      residual_.leap_time_;
  // rotational deceleration during land (rotates 0.25 pi)
  residual_.land_rot_acc_ =
      2 * (residual_.flight_rot_vel_ * residual_.land_time_ - mjPI / 4) /
      (residual_.land_time_ * residual_.land_time_);
}

// compute average foot position, depending on mode
void QuadrupedFlat::ResidualFn::AverageFootPos(
    double avg_foot_pos[3], double* foot_pos[kNumFoot]) const {
  if (current_mode_ == kModeBiped) {
    int handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
    if (handstand) {
      mju_add3(avg_foot_pos, foot_pos[kFootFL], foot_pos[kFootFR]);
    } else {
      mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    }
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
  } else {
    mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFL]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFR]);
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.25);
  }
}

// return phase as a function of time
double QuadrupedFlat::ResidualFn::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// horizontal Walk trajectory
void QuadrupedFlat::ResidualFn::Walk(double pos[2], double time) const {
  if (mju_abs(angvel_) < kMinAngvel) {
    // no rotation, go in straight line
    double forward[2] = {heading_[0], heading_[1]};
    mju_normalize(forward, 2);
    pos[0] = position_[0] + heading_[0] + time*speed_*forward[0];
    pos[1] = position_[1] + heading_[1] + time*speed_*forward[1];
  } else {
    // walk on a circle
    double angle = time * angvel_;
    double mat[4] = {mju_cos(angle), -mju_sin(angle),
                     mju_sin(angle),  mju_cos(angle)};
    mju_mulMatVec(pos, mat, heading_, 2, 2);
    pos[0] += position_[0];
    pos[1] += position_[1];
  }
}

// get gait
QuadrupedFlat::ResidualFn::A1Gait QuadrupedFlat::ResidualFn::GetGait() const {
  if (current_mode_ == kModeBiped)
    return kGaitTrot;
  return static_cast<A1Gait>(ReinterpretAsInt(current_gait_));
}

// return normalized target step height
double QuadrupedFlat::ResidualFn::StepHeight(double time, double footphase,
                                             double duty_ratio) const {
  double angle = fmod(time + mjPI - footphase, 2*mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

// compute target step height for all feet
void QuadrupedFlat::ResidualFn::FootStep(double step[kNumFoot], double time,
                                         A1Gait gait) const {
  double amplitude = parameters_[amplitude_param_id_];
  double duty_ratio = parameters_[duty_param_id_];
  for (A1Foot foot : kFootAll) {
    double footphase = 2*mjPI*kGaitPhase[gait][foot];
    step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
  }
}

// height during flip
double QuadrupedFlat::ResidualFn::FlipHeight(double time) const {
  if (time >= jump_time_ + flight_time_ + land_time_) {
    return kHeightQuadruped + ground_;
  }
  double h = 0;
  if (time < jump_time_) {
    h = kHeightQuadruped + time * crouch_vel_ + 0.5 * time * time * jump_acc_;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    h = kLeapHeight + jump_vel_*time - 0.5*9.81*time*time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    h = kLeapHeight - jump_vel_*time + 0.5*land_acc_*time*time;
  }
  return h + ground_;
}

// orientation during flip
//  total rotation = leap + flight + land
//            2*pi = pi/2 + 5*pi/4 + pi/4
void QuadrupedFlat::ResidualFn::FlipQuat(double quat[4], double time) const {
  double angle = 0;
  if (time >= jump_time_ + flight_time_ + land_time_) {
    angle = 2*mjPI;
  } else if (time >= crouch_time_ && time < jump_time_) {
    time -= crouch_time_;
    angle = 0.5 * jump_rot_acc_ * time * time + jump_rot_vel_ * time;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    angle = mjPI/2 + flight_rot_vel_ * time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    angle = 1.75*mjPI + flight_rot_vel_*time - 0.5*land_rot_acc_ * time * time;
  }
  int flip_dir = ReinterpretAsInt(parameters_[flip_dir_param_id_]);
  double axis[3] = {0, flip_dir ? 1.0 : -1.0, 0};
  mju_axisAngle2Quat(quat, axis, angle);
  mju_mulQuat(quat, orientation_, quat);
}


// --------------------- Residuals for quadruped task --------------------
//   Number of residuals: 4
//     Residual (0): position_z - average(foot position)_z - height_goal
//     Residual (1): position - goal_position
//     Residual (2): orientation - goal_orientation
//     Residual (3): control
//   Number of parameters: 1
//     Parameter (1): height_goal
// -----------------------------------------------------------------------
void QuadrupedHill::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  // ---------- Residual (0) ----------
  // standing height goal
  double height_goal = parameters_[0];

  // system's standing height
  double standing_height = SensorByName(model, data, "position")[2];

  // average foot height
  double FRz = SensorByName(model, data, "FR")[2];
  double FLz = SensorByName(model, data, "FL")[2];
  double RRz = SensorByName(model, data, "RR")[2];
  double RLz = SensorByName(model, data, "RL")[2];
  double avg_foot_height = 0.25 * (FRz + FLz + RRz + RLz);

  residual[0] = (standing_height - avg_foot_height) - height_goal;

  // ---------- Residual (1) ----------
  // goal position
  const double* goal_position = data->mocap_pos;

  // system's position
  double* position = SensorByName(model, data, "position");

  // position error
  mju_sub3(residual + 1, position, goal_position);

  // ---------- Residual (2) ----------
  // goal orientation
  double goal_rotmat[9];
  const double* goal_orientation = data->mocap_quat;
  mju_quat2Mat(goal_rotmat, goal_orientation);

  // system's orientation
  double body_rotmat[9];
  double* orientation = SensorByName(model, data, "orientation");
  mju_quat2Mat(body_rotmat, orientation);

  mju_sub(residual + 4, body_rotmat, goal_rotmat, 9);

  // ---------- Residual (3) ----------
  mju_copy(residual + 13, data->ctrl, model->nu);
}

// -------- Transition for quadruped task --------
//   If quadruped is within tolerance of goal ->
//   set goal to next from keyframes.
// -----------------------------------------------
void QuadrupedHill::TransitionLocked(mjModel* model, mjData* data) {
  // set mode to GUI selection
  if (mode > 0) {
    residual_.current_mode_ = mode - 1;
  } else {
    // ---------- Compute tolerance ----------
    // goal position
    const double* goal_position = data->mocap_pos;

    // goal orientation
    const double* goal_orientation = data->mocap_quat;

    // system's position
    double* position = SensorByName(model, data, "position");

    // system's orientation
    double* orientation = SensorByName(model, data, "orientation");

    // position error
    double position_error[3];
    mju_sub3(position_error, position, goal_position);
    double position_error_norm = mju_norm3(position_error);

    // orientation error
    double geodesic_distance =
        1.0 - mju_abs(mju_dot(goal_orientation, orientation, 4));

    // ---------- Check tolerance ----------
    double tolerance = 1.5e-1;
    if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
      // update task state
      residual_.current_mode_ += 1;
      if (residual_.current_mode_ == model->nkey) {
        residual_.current_mode_ = 0;
      }
    }
  }

  // ---------- Set goal ----------
  mju_copy3(data->mocap_pos, model->key_mpos + 3 * residual_.current_mode_);
  mju_copy4(data->mocap_quat, model->key_mquat + 4 * residual_.current_mode_);
}

}  // namespace mjpc
