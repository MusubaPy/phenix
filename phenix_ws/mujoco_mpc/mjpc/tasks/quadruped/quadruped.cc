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
#include <sstream>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

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

  double* torso_xmat = data->xmat + 9*torso_body_id_;
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  double* compos = SensorByName(model, data, "torso_subtreecom");
  if (!compos) return;


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
  if (!comvel) return;
  double* comacc = SensorByName(model, data, "torso_subtreelinacc");
  double capture_point[3];
  double fall_time = mju_sqrt(2*height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];

    UpdateGrfTarget(model, data, comvel, comacc);

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


  // ---------- Ground reaction forces ----------
  double foot_force_contact[kNumFoot][3];
  double foot_force[kNumFoot][3];
  double contact_point[kNumFoot][3];
  double contact_normal[kNumFoot][3];
  double max_normal_component[kNumFoot] = {0};
  bool foot_in_contact[kNumFoot] = {false};
  for (A1Foot foot : kFootAll) {
  mju_zero3(foot_force_contact[foot]);
  mju_zero3(foot_force[foot]);
    mju_zero3(contact_point[foot]);
    mju_zero3(contact_normal[foot]);
  }

  constexpr bool kEnableGrfDebugLog = false;

  [[maybe_unused]] std::ostringstream grf_log;
  if (kEnableGrfDebugLog) {
    grf_log.setf(std::ios::fixed, std::ios::floatfield);
    grf_log << std::setprecision(3);
  }
  [[maybe_unused]] const auto format_vec = [](const double v[3]) -> std::string {
    std::ostringstream vec_stream;
    vec_stream.setf(std::ios::fixed, std::ios::floatfield);
    vec_stream << std::setprecision(3)
               << "[" << std::setw(9) << v[0] << ", "
               << std::setw(9) << v[1] << ", "
               << std::setw(9) << v[2] << "]";
    return vec_stream.str();
  };

  const double grf_weight =
      (grf_cost_id_ >= 0 &&
       grf_cost_id_ < static_cast<int>(weight_.size()))
          ? weight_[grf_cost_id_]
          : 0.0;
  const double hind_weight = (hind_grf_cost_id_ >= 0 &&
                              hind_grf_cost_id_ <
                                  static_cast<int>(weight_.size()))
                                 ? weight_[hind_grf_cost_id_]
                                 : 0.0;

  auto weight_scale = [&](double weight_value) {
    if (weight_value <= 0.0) {
      return 0.0;
    }
    double denom = weight_value + kGrfWeightSoftReference;
    if (denom <= 0.0) {
      return 0.0;
    }
    double ratio = weight_value / denom;
    ratio = mju_clip(ratio, 0.0, 1.0);
    return std::sqrt(ratio);
  };

  const double grf_weight_scale = weight_scale(grf_weight);
  const double hind_weight_scale = weight_scale(hind_weight);
  const double net_force_weight_scale =
      std::max(grf_weight_scale, hind_weight_scale);

  for (int i = 0; i < data->ncon; ++i) {
    const mjContact& contact = data->contact[i];
    A1Foot contact_foot = kNumFoot;
    for (A1Foot foot : kFootAll) {
      if (contact.geom1 == foot_geom_id_[foot] ||
          contact.geom2 == foot_geom_id_[foot]) {
        contact_foot = foot;
        break;
      }
    }
    if (contact_foot == kNumFoot) continue;

    mjtNum force_local[6];
    mj_contactForce(model, data, i, force_local);
    double force_world[3];
    mju_mulMatVec(force_world, contact.frame, force_local, 3, 3);

    bool foot_is_geom1 = contact.geom1 == foot_geom_id_[contact_foot];
    double applied_force[3];
    if (foot_is_geom1) {
      mju_copy3(applied_force, force_world);
    } else {
      applied_force[0] = -force_world[0];
      applied_force[1] = -force_world[1];
      applied_force[2] = -force_world[2];
    }
  mju_addTo3(foot_force_contact[contact_foot], applied_force);

    double normal_world[3] = {contact.frame[6], contact.frame[7],
                              contact.frame[8]};
    if (foot_is_geom1) {
      normal_world[0] *= -1;
      normal_world[1] *= -1;
      normal_world[2] *= -1;
    }

    double normal_component = mju_abs(mju_dot3(applied_force, normal_world));
    if (normal_component > max_normal_component[contact_foot]) {
      max_normal_component[contact_foot] = normal_component;
      foot_in_contact[contact_foot] =
          normal_component > kContactForceThreshold;
      double normal_norm = mju_norm3(normal_world);
      if (normal_norm > kContactForceThreshold) {
        mju_scl3(contact_normal[contact_foot], normal_world,
                 1.0 / normal_norm);
      } else {
        mju_zero3(contact_normal[contact_foot]);
      }
      mju_copy3(contact_point[contact_foot], contact.pos);
    }
  }

  // prefer body-level contact wrench (`cfrc_ext`) for magnitude stability,
  // but fall back to accumulated contact forces if unavailable.
  for (A1Foot foot : kFootAll) {
    int geom_id = foot_geom_id_[foot];
    int body_id = (geom_id >= 0 && geom_id < model->ngeom)
                      ? model->geom_bodyid[geom_id]
                      : -1;
    if (body_id >= 0 && body_id < model->nbody) {
      const double* cfrc = data->cfrc_ext + 6 * body_id;
      // cfrc_ext stores wrench in rotation:translation order, grab the force.
      mju_copy3(foot_force[foot], cfrc + 3);
    }
  double contact_norm = mju_norm3(foot_force_contact[foot]);
  if (mju_norm3(foot_force[foot]) < kContactForceThreshold &&
    contact_norm > 0.0) {
      mju_copy3(foot_force[foot], foot_force_contact[foot]);
    }
  }

  const double gravity_norm = mju_norm3(model->opt.gravity);
  double foot_force_limit = total_mass_ * (gravity_norm + kDesiredAccelLimit);
  if (!std::isfinite(foot_force_limit) || foot_force_limit <= 0.0) {
    foot_force_limit = 1.0e3;
  }
  auto sanitize_force = [&](double force_vec[3]) {
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(force_vec[axis])) {
        force_vec[axis] = 0.0;
      }
    }
    double norm = mju_norm3(force_vec);
    if (!std::isfinite(norm)) {
      mju_zero3(force_vec);
      return;
    }
    if (norm > foot_force_limit && foot_force_limit > 1.0e-6) {
      mju_scl3(force_vec, force_vec, foot_force_limit / norm);
    }
  };
  for (A1Foot foot : kFootAll) {
    sanitize_force(foot_force[foot]);
  }

  double dt_grf = data->time - last_grf_update_time_;
  bool reset_grf_filter = (last_grf_update_time_ < 0.0) ||
                          (dt_grf <= 0.0) ||
                          (dt_grf > kDesiredAccelMaxDt);
  double alpha_grf = reset_grf_filter
                         ? 1.0
                         : 1.0 - std::exp(-dt_grf / kGrfFilterTimeConstant);
  alpha_grf = mju_clip(alpha_grf, 0.0, 1.0);
  double decay_grf = 1.0 - alpha_grf;

  bool effective_contact[kNumFoot] = {false};
  for (A1Foot foot : kFootAll) {
    if (foot_in_contact[foot]) {
      if (reset_grf_filter) {
        mju_copy3(filtered_foot_force_[foot], foot_force[foot]);
        mju_copy3(filtered_contact_normal_[foot], contact_normal[foot]);
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          filtered_foot_force_[foot][axis] +=
              alpha_grf *
              (foot_force[foot][axis] - filtered_foot_force_[foot][axis]);
          filtered_contact_normal_[foot][axis] +=
              alpha_grf *
              (contact_normal[foot][axis] -
               filtered_contact_normal_[foot][axis]);
        }
        double normal_norm = mju_norm3(filtered_contact_normal_[foot]);
        if (normal_norm > kContactForceThreshold) {
          mju_scl3(filtered_contact_normal_[foot],
                   filtered_contact_normal_[foot], 1.0 / normal_norm);
        } else {
          mju_zero3(filtered_contact_normal_[foot]);
        }
      }
    } else {
      if (reset_grf_filter) {
        mju_zero3(filtered_foot_force_[foot]);
        mju_zero3(filtered_contact_normal_[foot]);
      } else {
        mju_scl3(filtered_foot_force_[foot], filtered_foot_force_[foot],
                 decay_grf);
        if (mju_norm3(filtered_foot_force_[foot]) <
            0.5 * kSupportForceActivation) {
          mju_zero3(filtered_foot_force_[foot]);
          mju_zero3(filtered_contact_normal_[foot]);
        }
      }
    }
    sanitize_force(filtered_foot_force_[foot]);
    double filtered_norm = mju_norm3(filtered_foot_force_[foot]);
    effective_contact[foot] =
        foot_in_contact[foot] || filtered_norm > kSupportForceActivation;
  }
  last_grf_update_time_ = data->time;

  double net_grf[3] = {0, 0, 0};
  bool any_support = false;
  for (A1Foot foot : kFootAll) {
    mju_addTo3(net_grf, filtered_foot_force_[foot]);
    any_support = any_support || effective_contact[foot];
  }

  const int label_width = 8;
  const int value_width = 12;
  [[maybe_unused]] auto print_separator = [&]() {
    grf_log << '+' << std::string(label_width + 2, '-');
    for (int axis = 0; axis < 3; ++axis) {
      grf_log << '+' << std::string(value_width + 2, '-');
    }
    grf_log << "+\n";
  };

  if (kEnableGrfDebugLog) {
    grf_log << "\n[GRF] Таблица сил реакции опоры (Н)\n";
    print_separator();
    grf_log << "| " << std::setw(label_width) << std::left << "Нога    ";
    for (const char* axis_name : {"Fx", "Fy", "Fz"}) {
      grf_log << " | " << std::setw(value_width) << std::left << axis_name;
    }
    grf_log << " |\n";
    print_separator();
    for (A1Foot foot : kFootAll) {
      grf_log << "| " << std::setw(label_width) << std::left
              << kFootNames[foot];
      grf_log << std::right;
      for (int axis = 0; axis < 3; ++axis) {
        grf_log << " | " << std::setw(value_width) << foot_force[foot][axis];
      }
      grf_log << " |\n";
      grf_log << std::left;
    }
    print_separator();
  }

  const double vertical_capacity = total_mass_ * gravity_norm;

  double support_force = 0.0;
  for (A1Foot foot : kFootAll) {
    if (!effective_contact[foot]) continue;
    const double* force_eval = filtered_foot_force_[foot];
    double normal_eval[3];
    if (mju_norm3(filtered_contact_normal_[foot]) > kContactForceThreshold) {
      mju_copy3(normal_eval, filtered_contact_normal_[foot]);
    } else {
      mju_copy3(normal_eval, contact_normal[foot]);
    }
    double normal_norm = mju_norm3(normal_eval);
    if (normal_norm <= kContactForceThreshold) continue;
    mju_scl3(normal_eval, normal_eval, 1.0 / normal_norm);
    double normal_contrib = mju_dot3(force_eval, normal_eval);
    if (normal_contrib > 0.0) support_force += normal_contrib;
  }
  double support_fraction = 0.0;
  if (vertical_capacity > 1.0e-6) {
    support_fraction = mju_clip(support_force / vertical_capacity, 0.0, 1.0);
  }

  double planar_scale = 1.0;
  double planar_speed = 0.0;
  if (comvel != nullptr) {
    planar_speed = std::hypot(comvel[0], comvel[1]);
  }
  if (kPlanarSpeedHigh > kPlanarSpeedLow) {
    double speed_alpha =
        (planar_speed - kPlanarSpeedLow) /
        (kPlanarSpeedHigh - kPlanarSpeedLow);
    speed_alpha = mju_clip(speed_alpha, 0.0, 1.0);
    double speed_scale = 1.0 - (1.0 - kPlanarSpeedFloor) * speed_alpha;
    planar_scale = speed_scale;
  }
  if (gait == kGaitTrot || gait == kGaitCanter || gait == kGaitGallop) {
    planar_scale *= kDynamicGaitPlanarScale;
  }
  planar_scale = mju_clip(planar_scale, kNetForceMinScale, 1.0);

  double support_gate = support_fraction;
  if (kSupportResidualMinFraction > 1.0e-9) {
    support_gate /= kSupportResidualMinFraction;
  }
  support_gate = mju_clip(support_gate, 0.0, 1.0);
  double grf_target_effective[3];
  mju_copy3(grf_target_effective, grf_target_);
  mju_scl3(grf_target_effective, grf_target_effective,
           mju_clip(support_fraction, 0.0, 1.0));

  double net_force_error_raw[3];
  mju_sub3(net_force_error_raw, net_grf, grf_target_effective);

  double net_force_error_normalized[3] = {0, 0, 0};
  if (support_gate > 0.0 && net_force_weight_scale > 0.0) {
    double denom = vertical_capacity > 1.0 ? vertical_capacity : 1.0;
    for (int i = 0; i < 3; ++i) {
      net_force_error_normalized[i] = net_force_error_raw[i] / denom;
    }
    mju_scl3(net_force_error_normalized, net_force_error_normalized,
             support_gate);
    net_force_error_normalized[0] *= planar_scale;
    net_force_error_normalized[1] *= planar_scale;
    mju_scl3(net_force_error_normalized, net_force_error_normalized,
             net_force_weight_scale);
    mju_copy3(residual + counter, net_force_error_normalized);
  } else {
    mju_zero3(residual + counter);
  }
  counter += 3;

  if (kEnableGrfDebugLog) {
    grf_log << "Референсная net force: " << format_vec(grf_target_) << "\n";
    grf_log << "Фактическая net force:  " << format_vec(net_grf) << "\n";
    grf_log << "Ошибка net force (сырая): "
            << format_vec(net_force_error_raw) << "\n";
    grf_log << "Ошибка net force (норм.): "
            << format_vec(net_force_error_normalized) << "\n";
  }

  double hind_alignment_residual[kNumFoot] = {0, 0, 0, 0};
  for (A1Foot foot : kFootHind) {
    double alignment_residual = 0;
    if (hind_weight_scale <= 0.0) {
      hind_alignment_residual[foot] = 0.0;
      residual[counter++] = 0.0;
      continue;
    }
  [[maybe_unused]] std::ostringstream stage_log;
    if (kEnableGrfDebugLog) {
      stage_log.setf(std::ios::fixed, std::ios::floatfield);
      stage_log << std::setprecision(3);
      stage_log << "[GRF] Этапы выбора мотора (" << kFootNames[foot]
                << ")\n";
      stage_log << "  Контакт: "
                << (foot_in_contact[foot] ? "да" : "нет") << "\n";
    }
    if (foot_in_contact[foot] && effective_contact[foot]) {
      double force_norm = mju_norm3(filtered_foot_force_[foot]);
      if (kEnableGrfDebugLog) {
        stage_log << "  Норма силы: " << force_norm << "\n";
      }
      if (force_norm > kContactForceThreshold) {
        double grf_unit[3];
        mju_copy3(grf_unit, filtered_foot_force_[foot]);
        mju_scl3(grf_unit, grf_unit, 1.0 / force_norm);
        if (kEnableGrfDebugLog) {
          stage_log << "  Единичный GRF: " << format_vec(grf_unit) << "\n";
        }
        double motor_vec[3];
        bool has_motor =
            MotorVector(foot, contact_point[foot], contact_normal[foot], data,
                         motor_vec);
        if (has_motor) {
          if (kEnableGrfDebugLog) {
            stage_log << "  Вектор к мотору: " << format_vec(motor_vec)
                      << "\n";
          }
          double plane_normal[3];
          bool has_plane = MotorPlaneNormal(foot, data, plane_normal);
          if (kEnableGrfDebugLog) {
            stage_log << "  Нормаль плоскости моторов: ";
          }
          if (has_plane) {
            if (kEnableGrfDebugLog) {
              stage_log << format_vec(plane_normal) << "\n";
            }
            double grf_proj[3];
            mju_copy3(grf_proj, grf_unit);
            double grf_dot = mju_dot3(grf_proj, plane_normal);
            mju_addToScl3(grf_proj, plane_normal, -grf_dot);
            double grf_proj_norm = mju_norm3(grf_proj);
            if (kEnableGrfDebugLog) {
              stage_log << "  Проекция GRF: ";
            }
            if (grf_proj_norm > kContactForceThreshold) {
              mju_scl3(grf_proj, grf_proj, 1.0 / grf_proj_norm);
              if (kEnableGrfDebugLog) {
                stage_log << format_vec(grf_proj) << "\n";
              }
              double motor_proj[3];
              mju_copy3(motor_proj, motor_vec);
              double motor_dot = mju_dot3(motor_proj, plane_normal);
              mju_addToScl3(motor_proj, plane_normal, -motor_dot);
              double motor_proj_norm = mju_norm3(motor_proj);
              if (kEnableGrfDebugLog) {
                stage_log << "  Проекция мотора: ";
              }
              if (motor_proj_norm > kContactForceThreshold) {
                mju_scl3(motor_proj, motor_proj, 1.0 / motor_proj_norm);
                if (kEnableGrfDebugLog) {
                  stage_log << format_vec(motor_proj) << "\n";
                }
                double cross[3];
                mju_cross(cross, grf_proj, motor_proj);
                alignment_residual = mju_norm3(cross);
                // масштабируем штраф по силе контакта, чтобы слабые касания не ломали оптимизацию
                double contact_scale =
                    mju_tanh(force_norm / (vertical_capacity * 0.25 + 1.0e-6));
                alignment_residual *= contact_scale;
                if (kEnableGrfDebugLog) {
                  stage_log << "  Векторное произведение: "
                            << format_vec(cross) << "\n";
                }
              } else {
                if (kEnableGrfDebugLog) {
                  stage_log << "недостаточная норма (" << motor_proj_norm
                            << ")\n";
                }
              }
            } else {
              if (kEnableGrfDebugLog) {
                stage_log << "недостаточная норма (" << grf_proj_norm
                          << ")\n";
              }
            }
          } else {
            if (kEnableGrfDebugLog) {
              stage_log << "не найдена\n";
            }
          }
        } else {
          if (kEnableGrfDebugLog) {
            stage_log << "  Вектор к мотору: не найден\n";
          }
        }
      } else {
        if (kEnableGrfDebugLog) {
          stage_log << "  Норма силы ниже порога (" << kContactForceThreshold
                    << ")\n";
        }
      }
    } else {
      if (kEnableGrfDebugLog) {
        stage_log << "  Контакт отсутствует\n";
      }
    }
    if (kEnableGrfDebugLog) {
      stage_log << "  Итоговый residual: " << alignment_residual << "\n";
      grf_log << stage_log.str();
    }
    double hind_scale = mju_max(planar_scale, kHindAlignmentMinScale);
    alignment_residual *= hind_scale;
    alignment_residual *= hind_weight_scale;
    hind_alignment_residual[foot] = alignment_residual;
    residual[counter++] = alignment_residual;
  }

  bool debug_logging = false;
  if (debug_log_param_id_ >= 0 &&
      debug_log_param_id_ < static_cast<int>(parameters_.size())) {
    debug_logging = parameters_[debug_log_param_id_] > 0.5;
  }
  if (debug_logging) {
    double interval = kDebugLogInterval;
    if (last_debug_print_time_ < 0.0 ||
        data->time - last_debug_print_time_ >= interval) {
      last_debug_print_time_ = data->time;
      static const char* const kModeNames[] = {
          "Quadruped", "Biped", "Walk", "Scramble", "Flip"};
      static const char* const kGaitNames[] = {
          "Stand", "Walk", "Trot", "Canter", "Gallop"};
      const char* mode_name =
          (current_mode_ >= 0 && current_mode_ < kNumMode)
              ? kModeNames[current_mode_]
              : "Unknown";
      const char* gait_name = (gait >= 0 && gait < kNumGait)
                                  ? kGaitNames[gait]
                                  : "Unknown";
      std::ostringstream dbg;
      dbg.setf(std::ios::fixed, std::ios::floatfield);
      dbg << std::setprecision(3);
      dbg << "[GRF DEBUG] t=" << data->time << " mode=" << mode_name
          << " gait=" << gait_name << " planar_speed=" << planar_speed
          << " support=" << support_fraction
          << " planar_scale=" << planar_scale
          << " weight(GRF)=" << grf_weight
      << " weight_scale(GRF)=" << grf_weight_scale
      << " weight(HindAlign)=" << hind_weight
      << " weight_scale(HindAlign)=" << hind_weight_scale << "\n";
      dbg << "  target=" << format_vec(grf_target_)
          << " net=" << format_vec(net_grf)
          << " error=" << format_vec(net_force_error_raw) << "\n";

  double weighted_terms[mjpc::kMaxCostTerms];
  double raw_terms[mjpc::kMaxCostTerms];
  std::fill(weighted_terms, weighted_terms + mjpc::kMaxCostTerms, 0.0);
  std::fill(raw_terms, raw_terms + mjpc::kMaxCostTerms, 0.0);
      CostTerms(weighted_terms, residual, /*weighted=*/true);
      CostTerms(raw_terms, residual, /*weighted=*/false);
      double total_objective = 0.0;
      for (int term = 0; term < num_term_; ++term) {
        if (std::isfinite(weighted_terms[term])) {
          total_objective += weighted_terms[term];
        }
      }
      auto cost_term_name = [&](int term_index) -> std::string {
        if (term_index >= 0 &&
            term_index < static_cast<int>(task_->weight_names.size()) &&
            !task_->weight_names[term_index].empty()) {
          return task_->weight_names[term_index];
        }
        std::ostringstream name_stream;
        name_stream << "term[" << term_index << "]";
        return name_stream.str();
      };
      dbg << "  objective_total=" << total_objective << "\n";
      dbg << "  cost_impact:";
      if (num_term_ == 0) {
        dbg << " <no cost terms detected>";
      }
      dbg << "\n";
      for (int term = 0; term < num_term_; ++term) {
        double weighted_cost = weighted_terms[term];
        double raw_cost = raw_terms[term];
        double weight_value =
            (term >= 0 && term < static_cast<int>(weight_.size()))
                ? weight_[term]
                : 0.0;
        if (!std::isfinite(weighted_cost) && !std::isfinite(raw_cost)) {
          continue;
        }
        if (!std::isfinite(weighted_cost)) {
          weighted_cost = 0.0;
        }
        if (!std::isfinite(raw_cost)) {
          raw_cost = 0.0;
        }
        double share = 0.0;
        if (total_objective > 1.0e-9) {
          share = weighted_cost / total_objective;
        }
        dbg << "    " << cost_term_name(term)
            << ": weighted=" << weighted_cost
            << " raw=" << raw_cost
            << " weight=" << weight_value
            << " share=" << share * 100.0 << "%\n";
      }
      for (A1Foot foot : kFootAll) {
        double force_norm = mju_norm3(filtered_foot_force_[foot]);
        dbg << "  " << kFootNames[foot]
            << " contact=" << (effective_contact[foot] ? "Y" : "N")
            << " raw_contact=" << (foot_in_contact[foot] ? "Y" : "N")
            << " |f|=" << force_norm
            << " f=" << format_vec(filtered_foot_force_[foot]) << "\n";
      }
      for (A1Foot foot : kFootHind) {
        dbg << "    hind_align[" << kFootNames[foot]
            << "]=" << hind_alignment_residual[foot] << "\n";
      }
      dbg << std::flush;
      std::cout << dbg.str();
    }
  }

  if (kEnableGrfDebugLog) {
    std::cout << grf_log.str();
  }


  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

bool QuadrupedFlat::ResidualFn::MotorVector(A1Foot foot,
                                            const double contact_point[3],
                                            const double contact_normal[3],
                                            const mjData* data,
                                            double motor_vector[3]) const {
  if (!contact_point || !contact_normal) return false;
  if (mju_norm3(contact_normal) < kContactForceThreshold) return false;
  double best_alignment = -1.0;
  bool found = false;
  double contact_to_motor[3];
  double normalized_motor[3];
  for (int motor = 0; motor < kMotorsPerFoot; ++motor) {
    int id = motor_body_id_[foot][motor];
    if (id < 0) continue;
    const double* motor_pos = data->xipos + 3 * id;
    mju_sub3(contact_to_motor, motor_pos, contact_point);
    double length = mju_norm3(contact_to_motor);
    if (length <= 1.0e-9) continue;
    mju_scl3(normalized_motor, contact_to_motor, 1.0 / length);
    double alignment = mju_abs(mju_dot3(normalized_motor, contact_normal));
    if (alignment > best_alignment) {
      best_alignment = alignment;
      mju_copy3(motor_vector, normalized_motor);
      found = true;
    }
  }
  return found;
}

bool QuadrupedFlat::ResidualFn::MotorPlaneNormal(A1Foot foot,
                                                 const mjData* data,
                                                 double plane_normal[3]) const {
  if (!plane_normal) return false;
  int body_id = motor_body_id_[foot][1];  // default to thigh
  if (body_id < 0) {
    body_id = motor_body_id_[foot][0];  // fallback hip
  }
  if (body_id < 0) {
    body_id = motor_body_id_[foot][2];  // fallback calf
  }
  if (body_id < 0) return false;
  const double* xmat = data->xmat + 9 * body_id;
  plane_normal[0] = xmat[1];
  plane_normal[1] = xmat[4];
  plane_normal[2] = xmat[7];
  double norm = mju_norm3(plane_normal);
  if (norm < kContactForceThreshold) {
    return false;
  }
  mju_scl3(plane_normal, plane_normal, 1.0 / norm);
  return true;
}

void QuadrupedFlat::ResidualFn::UpdateGrfTarget(const mjModel* model,
                                                const mjData* data,
                                                const double* comvel,
                                                const double* comacc) const {
  if (total_mass_ <= 0) {
    mju_zero3(grf_target_);
    return;
  }

  double base_target[3];
  mju_copy3(base_target, model->opt.gravity);
  mju_scl3(base_target, base_target, -total_mass_);
  mju_copy3(grf_target_, base_target);

  double measured_accel[3] = {0, 0, 0};
  bool have_measurement = false;

  if (comacc != nullptr) {
    mju_copy3(measured_accel, comacc);
    have_measurement = true;
  }

  double dt = data->time - last_accel_time_;
  bool reset_filter = (last_accel_time_ < 0.0) || (dt <= 0.0) ||
                      (dt > kDesiredAccelMaxDt);

  if (!have_measurement && comvel != nullptr) {
    if (!reset_filter) {
      mju_sub3(measured_accel, comvel, last_com_vel_);
      if (dt > 1.0e-6) {
        mju_scl3(measured_accel, measured_accel, 1.0 / dt);
        have_measurement = true;
      } else {
        reset_filter = true;
      }
    }
  }

  if (!have_measurement) {
    if (reset_filter) {
      mju_zero3(filtered_desired_accel_);
    }
    if (comvel != nullptr) {
      mju_copy3(last_com_vel_, comvel);
    }
    last_accel_time_ = data->time;
    return;
  }

  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(measured_accel[i])) {
      measured_accel[i] = 0.0;
    }
    measured_accel[i] = mju_clip(measured_accel[i],
                                 -kDesiredAccelLimit, kDesiredAccelLimit);
  }

  if (reset_filter) {
    mju_copy3(filtered_desired_accel_, measured_accel);
  } else {
    double alpha = 1.0 - std::exp(-dt / kDesiredAccelFilterTimeConstant);
    alpha = mju_min(1.0, mju_max(0.0, alpha));
    for (int i = 0; i < 3; ++i) {
      filtered_desired_accel_[i] +=
          alpha * (measured_accel[i] - filtered_desired_accel_[i]);
    }
  }

  mju_addToScl3(grf_target_, filtered_desired_accel_, total_mass_);

  double gravity_norm = mju_norm3(model->opt.gravity);
  double max_force_norm = total_mass_ * (gravity_norm + kDesiredAccelLimit);
  double current_norm = mju_norm3(grf_target_);
  if (max_force_norm > 0 && current_norm > max_force_norm) {
    mju_scl3(grf_target_, grf_target_, max_force_norm / current_norm);
  }

  double horizontal_limit = total_mass_ * kDesiredAccelLimit;
  grf_target_[0] = mju_clip(grf_target_[0], -horizontal_limit, horizontal_limit);
  grf_target_[1] = mju_clip(grf_target_[1], -horizontal_limit, horizontal_limit);
  double vertical_limit = total_mass_ * (gravity_norm + kDesiredAccelLimit);
  grf_target_[2] = mju_clip(grf_target_[2], -vertical_limit, vertical_limit);

  if (comvel != nullptr) {
    mju_copy3(last_com_vel_, comvel);
  }
  last_accel_time_ = data->time;
}

//  ============  transition  ============
void QuadrupedFlat::TransitionLocked(mjModel* model, mjData* data) {
  // Ensure parameter/body ids are initialized (defensive against early calls)
  if (residual_.cadence_param_id_ < 0 ||
      residual_.gait_param_id_ < 0 ||
      residual_.gait_switch_param_id_ < 0 ||
      residual_.torso_body_id_ < 0 ||
      residual_.head_site_id_ < 0 ||
      residual_.goal_mocap_id_ < 0) {
    ResetLocked(model);
  }
  // ---------- handle mjData reset ----------
  if (data->time < residual_.last_transition_time_ ||
      residual_.last_transition_time_ == -1) {
    if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
      mode = ResidualFn::kModeQuadruped;  // mode stateful, switch to Quadruped
    }
    if (residual_.gait_switch_param_id_ >= 0 &&
        residual_.gait_switch_param_id_ < static_cast<int>(parameters.size())) {
      int current_switch =
          ReinterpretAsInt(parameters[residual_.gait_switch_param_id_]);
      if (current_switch == 0) {
        parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(1);
      }
    }
    residual_.last_transition_time_ = residual_.phase_start_time_ =
        residual_.phase_start_ = data->time;
    // initialize current_gait_ from the parameter on first reset
    if (residual_.gait_param_id_ >= 0 &&
        residual_.gait_param_id_ < static_cast<int>(parameters.size())) {
      residual_.current_gait_ = parameters[residual_.gait_param_id_];
    } else {
      residual_.current_gait_ = ResidualFn::kGaitStand;
    }
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
  double phase_velocity = residual_.phase_velocity_;
  if (residual_.cadence_param_id_ >= 0 &&
      residual_.cadence_param_id_ < static_cast<int>(parameters.size())) {
    phase_velocity = 2 * mjPI * parameters[residual_.cadence_param_id_];
  }
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
  int auto_switch = 0;
  if (residual_.gait_switch_param_id_ >= 0 &&
      residual_.gait_switch_param_id_ < static_cast<int>(parameters.size())) {
    auto_switch = ReinterpretAsInt(parameters[residual_.gait_switch_param_id_]);
  }
  if (mode == ResidualFn::kModeBiped) {
    // biped always trots
    if (residual_.gait_param_id_ >= 0 &&
        residual_.gait_param_id_ < static_cast<int>(parameters.size())) {
      parameters[residual_.gait_param_id_] =
          ReinterpretAsDouble(ResidualFn::kGaitTrot);
    }
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
        if (residual_.gait_param_id_ >= 0 &&
            residual_.gait_param_id_ < static_cast<int>(parameters.size())) {
          parameters[residual_.gait_param_id_] = ReinterpretAsDouble(gait);
        }
        residual_.gait_switch_time_ = data->time;
      }
    }
  }


  // ---------- handle gait switch, manual or auto ----------
  double gait_selection = residual_.current_gait_;
  if (residual_.gait_param_id_ >= 0 &&
      residual_.gait_param_id_ < static_cast<int>(parameters.size())) {
    gait_selection = parameters[residual_.gait_param_id_];
  }
  if (gait_selection != residual_.current_gait_) {
    residual_.current_gait_ = gait_selection;
    ResidualFn::A1Gait gait = residual_.GetGait();
    auto set_param = [&](int idx, double val) {
      if (idx >= 0 && idx < static_cast<int>(parameters.size())) {
        parameters[idx] = val;
      }
    };
    auto set_weight = [&](int idx, double val) {
      if (idx >= 0 && idx < static_cast<int>(weight.size())) {
        weight[idx] = val;
      }
    };
    set_param(residual_.duty_param_id_, ResidualFn::kGaitParam[gait][0]);
    set_param(residual_.cadence_param_id_, ResidualFn::kGaitParam[gait][1]);
    set_param(residual_.amplitude_param_id_, ResidualFn::kGaitParam[gait][2]);
    set_weight(residual_.balance_cost_id_, ResidualFn::kGaitParam[gait][3]);
    set_weight(residual_.upright_cost_id_, ResidualFn::kGaitParam[gait][4]);
    set_weight(residual_.height_cost_id_, ResidualFn::kGaitParam[gait][5]);
  }


  // ---------- Walk ----------
  if (residual_.goal_mocap_id_ < 0 ||
      3 * residual_.goal_mocap_id_ + 2 >= model->nmocap * 3) {
    return;  // invalid goal mocap index
  }
  double* goal_pos = data->mocap_pos + 3*residual_.goal_mocap_id_;
  if (mode == ResidualFn::kModeWalk) {
  int turn_idx = ParameterIndex(model, "Walk turn");
  int speed_idx = ParameterIndex(model, "Walk speed");
  double angvel = (turn_idx >= 0 && turn_idx < static_cast<int>(parameters.size())) ?
          parameters[turn_idx] : 0;
  double speed = (speed_idx >= 0 && speed_idx < static_cast<int>(parameters.size())) ?
           parameters[speed_idx] : 0;

    // current torso direction
  if (residual_.torso_body_id_ < 0) return;
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
  if (!compos) return;
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
      auto set_w = [&](const char* name, double val){
        int idx = CostTermByName(model, name);
        if (idx >= 0 && idx < static_cast<int>(weight.size())) weight[idx] = val;
      };
      set_w("Upright", 0.2);
      set_w("Height", 5);
      set_w("Position", 0);
      set_w("Gait", 0);
      set_w("Balance", 0);
      set_w("Effort", 0.005);
      set_w("Posture", 0.1);
      if (residual_.gait_switch_param_id_ >= 0 &&
          residual_.gait_switch_param_id_ < static_cast<int>(parameters.size())) {
        parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(0);
      }
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
  residual_.debug_log_param_id_ = ParameterIndex(model, "Debug GRF log");
  residual_.balance_cost_id_ = CostTermByName(model, "Balance");
  residual_.upright_cost_id_ = CostTermByName(model, "Upright");
  residual_.height_cost_id_ = CostTermByName(model, "Height");
  residual_.grf_cost_id_ = CostTermByName(model, "GRF");
  residual_.hind_grf_cost_id_ = CostTermByName(model, "Hind GRF Align");

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

  // motor body ids by foot
  for (int foot = 0; foot < ResidualFn::kNumFoot; ++foot) {
    for (int motor = 0; motor < ResidualFn::kMotorsPerFoot; ++motor) {
      const char* primary = ResidualFn::kMotorBodyNames[foot][motor];
      int body_id = -1;
      if (primary) {
        body_id = mj_name2id(model, mjOBJ_BODY, primary);
      }
      if (body_id < 0) {
        const char* fallback = ResidualFn::kMotorBodyFallback[foot][motor];
        if (fallback) {
          body_id = mj_name2id(model, mjOBJ_BODY, fallback);
        }
      }
      if (body_id < 0) {
        if (primary) {
          mju_error_s("body '%s' not found", primary);
        } else {
          mju_error("motor body name missing");
        }
      }
      residual_.motor_body_id_[foot][motor] = body_id;
    }
  }

  // compute total mass (robot subtree only) and GRF target
  residual_.total_mass_ = 0;
  if (residual_.torso_body_id_ >= 0 &&
      residual_.torso_body_id_ < model->nbody) {
    int root_id = model->body_rootid[residual_.torso_body_id_];
    for (int i = 0; i < model->nbody; ++i) {
      if (model->body_rootid[i] == root_id) {
        residual_.total_mass_ += model->body_mass[i];
      }
    }
  }
  mju_zero3(residual_.grf_target_);
  if (residual_.total_mass_ > 0) {
    for (int i = 0; i < 3; ++i) {
      residual_.grf_target_[i] = -residual_.total_mass_ * model->opt.gravity[i];
    }
  }
  mju_zero3(residual_.filtered_desired_accel_);
  mju_zero3(residual_.last_com_vel_);
  residual_.last_accel_time_ = -1.0;
  residual_.last_debug_print_time_ = -1.0;

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
  int gi = ReinterpretAsInt(current_gait_);
  // Fallback: if reinterpret produced invalid value, try numeric cast
  if (gi < 0 || gi >= kNumGait) {
    // try to round the double value into an int index within [0, kNumGait)
    int rounded = static_cast<int>(current_gait_ >= 0 ? current_gait_ + 0.5
                                                      : current_gait_ - 0.5);
    if (rounded >= 0 && rounded < kNumGait) {
      gi = rounded;
    } else {
      gi = kGaitStand;
    }
  }
  // Clamp to valid range
  if (gi < 0) gi = 0;
  if (gi >= kNumGait) gi = kNumGait - 1;
  return static_cast<A1Gait>(gi);
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
  // guard against invalid gait
  int gi = static_cast<int>(gait);
  if (gi < 0 || gi >= kNumGait) {
    gi = kGaitStand;
  }
  for (A1Foot foot : kFootAll) {
    double footphase = 2*mjPI*kGaitPhase[gi][foot];
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
