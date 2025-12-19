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

#ifndef MJPC_TASKS_QUADRUPED_MOD_QUADRUPED_H_
#define MJPC_TASKS_QUADRUPED_MOD_QUADRUPED_H_

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

// Сведения о контактах лап для логгера/выравнивания GRF.
struct FootContactInfo {
  FootContactInfo() {
    mju_zero3(force);
    mju_zero3(normal);
    mju_zero3(point);
    weight = 0.0;
    in_contact = false;
  }
  double force[3];
  double normal[3];
  double point[3];
  double weight;
  bool in_contact;
};

class QuadrupedFlatMod : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
  explicit ResidualFn(const QuadrupedFlatMod* task)
    : mjpc::BaseResidualFn(task), debug_log_state_(std::make_shared<DebugLogState>()) {}
    ResidualFn(const ResidualFn&) = default;
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class QuadrupedFlatMod;
    //  ============  enums  ============
    // modes
    enum A1Mode {
      kModeQuadruped = 0,
      kModeBiped,
      kModeWalk,
      kModeScramble,
      kModeFlip,
      kNumMode
    };

    // feet
    enum A1Foot {
      kFootFL  = 0,
      kFootHL,
      kFootFR,
      kFootHR,
      kNumFoot
    };

    // gaits
    enum A1Gait {
      kGaitStand = 0,
      kGaitWalk,
      kGaitTrot,
      kGaitCanter,
      kGaitGallop,
      kNumGait
    };

    //  ============  constants  ============
    constexpr static A1Foot kFootAll[kNumFoot] = {kFootFL, kFootHL,
                                                  kFootFR, kFootHR};
    constexpr static A1Foot kFootHind[2] = {kFootHL, kFootHR};
  constexpr static A1Foot kFootFront[2] = {kFootFL, kFootFR};
    constexpr static A1Gait kGaitAll[kNumGait] = {kGaitStand, kGaitWalk,
                                                  kGaitTrot, kGaitCanter,
                                                  kGaitGallop};
    constexpr static const char* kAbductionJointNames[kNumFoot] = {
        "FL_hip_joint", "RL_hip_joint", "FR_hip_joint", "RR_hip_joint"};
    constexpr static const char* kHipJointNames[kNumFoot] = {
        "FL_thigh_joint", "RL_thigh_joint", "FR_thigh_joint",
        "RR_thigh_joint"};
    constexpr static const char* kKneeJointNames[kNumFoot] = {
        "FL_calf_joint", "RL_calf_joint", "FR_calf_joint", "RR_calf_joint"};

    // gait phase signature (normalized)
    constexpr static double kGaitPhase[kNumGait][kNumFoot] =
    {
    // FL     HL     FR     HR
      {0,     0,     0,     0   },   // stand
      {0,     0.75,  0.5,   0.25},   // walk
      {0,     0.5,   0.5,   0   },   // trot
      {0,     0.33,  0.33,  0.66},   // canter
      {0,     0.4,   0.05,  0.35}    // gallop
    };

    // gait parameters, set when switching into gait
    constexpr static double kGaitParam[kNumGait][6] =
    {
    // duty ratio  cadence  amplitude  balance   upright   height
    // unitless    Hz       meter      unitless  unitless  unitless
      {1,          1,       0,         0,        1,        1},      // stand
      {0.75,       1,       0.03,      0,        1,        1},      // walk
      {0.45,       2,       0.03,      0.2,      1,        1},      // trot
      {0.4,        4,       0.05,      0.03,     0.5,      0.2},    // canter
      {0.3,        3.5,     0.10,      0.03,     0.2,      0.1}     // gallop
    };

    // velocity ranges for automatic gait switching, meter/second
    constexpr static double kGaitAuto[kNumGait] =
    {
      0,     // stand
      0.02,  // walk
      0.02,  // trot
      0.6,   // canter
      2,     // gallop
    };
    // notes:
    // - walk is never triggered by auto-gait
    // - canter actually has a wider range than gallop

    // automatic gait switching: time constant for com speed filter
    constexpr static double kAutoGaitFilter = 0.2;    // second

    // automatic gait switching: minimum time between switches
    constexpr static double kAutoGaitMinTime = 1;     // second

    // target torso height over feet when quadrupedal
    constexpr static double kHeightQuadruped = 0.25;  // meter

    // target torso height over feet when bipedal
    constexpr static double kHeightBiped = 0.6;       // meter

    // radius of foot geoms
    constexpr static double kFootRadius = 0.02;       // meter

    // below this target yaw velocity, walk straight
    constexpr static double kMinAngvel = 0.01;        // radian/second

    // posture gain factors for abduction, hip, knee
    constexpr static double kJointPostureGain[3] = {2, 1, 1};  // unitless

    // flip: crouching height, from which leap is initiated
    constexpr static double kCrouchHeight = 0.15;     // meter

    // flip: leap height, beginning of flight phase
    constexpr static double kLeapHeight = 0.5;        // meter

    // flip: maximum height of flight phase
    constexpr static double kMaxHeight = 0.8;         // meter

    //  ============  methods  ============
    // return internal phase clock
    double GetPhase(double time) const;

    // return current gait
    A1Gait GetGait() const;

    // compute average foot position, depending on mode
    void AverageFootPos(double avg_foot_pos[3],
                        double* foot_pos[kNumFoot]) const;

    // return normalized target step height
    double StepHeight(double time, double footphase, double duty_ratio) const;

    // compute target step height for all feet
    void FootStep(double step[kNumFoot], double time, A1Gait gait) const;

    // walk horizontal position given time
    void Walk(double pos[2], double time) const;

    // height during flip
    double FlipHeight(double time) const;

    // orientation during flip
    void FlipQuat(double quat[4], double time) const;

    //  ============  task state variables, managed by Transition  ============
    A1Mode current_mode_       = kModeQuadruped;
    double last_transition_time_ = -1;

    // common mode states
    double mode_start_time_  = 0;
    double position_[3]       = {0};

    // walk states
    double heading_[2]        = {0};
    double speed_             = 0;
    double angvel_            = 0;

    // backflip states
    double ground_            = 0;
    double orientation_[4]    = {0};
    double save_gait_switch_  = 0;
    std::vector<double> save_weight_;

    // === MOD: GRF tuning parameters (set via env vars in ResetLocked) ===
    // The following fields are the core experimental knobs introduced in
    // the 'mod' task. Keep them grouped to simplify audits and sweeps.
    // Env variables: MJPC_GRF_PER_FOOT_SCALE, MJPC_MASS_DISTRIBUTION,
    // MJPC_GRF_WEIGHT, MJPC_GRF_HIND_WEIGHT, MJPC_GRF_FRONT_WEIGHT,
    // MJPC_GRF_MOTOR_BLEND, MJPC_DISABLE_MOTOR_BLEND,
    // MJPC_CONTACT_STABLE_STEPS, MJPC_REFLECT_NET_FORCE_GAIN
    // Default per-foot GRF scaling (match sweep defaults used by scripts).
    double grf_per_foot_scale_[4] = {0.8, 0.8, 1.2, 1.2};
    bool grf_normalize_ = false;
    double grf_loss_mix_ = 0.0;
    // Default: no transition boost for quick checks.
    double grf_transition_boost_ = 0.0;
    double internal_grf_align_weight_ = 0.001;
      // Target joint selection / Kuznetsov mode parameters
      // MJPC_GRF_TARGET_MODE: kuznetsov | alexander | blend
      std::string grf_target_mode_ = "kuznetsov";
      // Defaults chosen to match the quick-check smoothing candidate
      // (tau=1.0, alpha=0.6) used by our sweep scripts.
      double target_blend_alpha_ = 0.6;  // MJPC_TARGET_BLEND_ALPHA
      double target_smooth_tau_ = 1.0;   // MJPC_TARGET_SMOOTH_TAU (seconds)

      // Joint fixation penalty
      // Default fixation weight used for quick checks (small, non-zero).
      double fixation_weight_ = 1e-4;     // MJPC_FIXATION_WEIGHT
      double fixation_threshold_ = 0.0;  // MJPC_FIXATION_THRESHOLD (unused: future)

      // Power penalty (p = torque * joint_vel). Value added into effort residuals
      double power_penalty_weight_ = 0.0;  // MJPC_POWER_PENALTY_WEIGHT
      std::string power_penalty_mode_ = "l2"; // 'l2' or 'l1'
      // scaling knob to amplify the built-in Height cost (1.0 = no change)
      double height_weight_scale_ = 1.0;  // MJPC_HEIGHT_WEIGHT_SCALE

      // Optional biarticular coupling gain (experimental)
      double biarticular_gain_ = 0.0;    // MJPC_BIARTICULAR_GAIN
      // motor selection blending beta (softmax). >0 enables blending
      double grf_motor_blend_beta_ = 0.0;
    // contact stability steps (can be overridden via env MJPC_CONTACT_STABLE_STEPS)
    // Default contact stability required steps (match sweep default 10).
    int contact_stable_steps_ = 10;
    // Hardcoded testing weights for quick validation (can be overridden by env)
    // Made conservative by default to avoid destabilizing early runs.
    // Conservative experimental defaults (tiny values so these terms are
    // effectively inactive unless explicitly increased).
    double grf_hind_weight_ = 1e-7;   // scale for hind alignment residual
    double grf_front_weight_ = 1e-7; // scale for front alignment residual
    // Optional horizontal GRF penalty (default off). Gate carefully; may
    // destabilize if enabled with large weights.
    double grf_horiz_weight_ = 0.0;  // MJPC_GRF_HORIZ_WEIGHT
    // Per-foot maximum residual magnitude to avoid exploding objective
    // contributions when weights are tuned aggressively.
    double grf_max_residual_ = 0.5;  // MJPC_GRF_MAX_RESIDUAL (meters or unitless vector norm)

    // NOTE: runtime sensor scaling removed — measured GRF are used as-is.
    // Historically we allowed scaling measured GRF at runtime (MJPC_GRF_SENSOR_SCALE)
    // for quick experiments; this was found to mask real energy effects and is
    // therefore disabled. If you need to re-enable it, add a careful
    // experiment flag here and document it.

    // contact stability steps overridden/stored in the ResidualFn instance.
    // (see ResidualFn::contact_stable_steps_)

    // optional net-force reflection gain. If >0, front mirrored references
    // are nudged by a scaled negative net force to encourage net force -> [0,0,mg].
    double reflect_net_force_gain_ = 0.0;

    // convenience flag: when MJPC_GRF_WEIGHT is present, both hind/front
    // weights will be set to that scalar value (see ResetLocked).
    // Future: support direct reading from XML numeric field when explicitly
    // requested via MJPC_USE_XML_GRF_WEIGHT.
    bool use_xml_grf_weight_ = false;
    // gait-related states
    double current_gait_      = kGaitStand;
    double phase_start_       = 0;
    double phase_start_time_  = 0;
    double phase_velocity_    = 0;
    double com_vel_[2]        = {0};
    double gait_switch_time_  = 0;

    // (Warmup/measurement gating removed — logging now always allowed)

    // startup hold then auto-walk
    double startup_hold_duration_ = 6.0;  // stand still duration
    double startup_begin_time_ = 0.0;
    bool startup_walk_triggered_ = false;
    double startup_walk_time_ = 0.0;      // when we switched to walk
    double startup_auto_delay_ = 0.5;     // delay before reenabling auto gait
    double startup_ramp_duration_ = 2.0;  // seconds to ramp desired height
    double startup_height_scale_ = 0.0;   // 0..1 multiplier for height goal
    double startup_height_start_ = 0.0;   // CoM height at ramp start
    bool startup_height_captured_ = false;

    //  ============  constants, computed in Reset()  ============
    int torso_body_id_        = -1;
    int head_site_id_         = -1;
    int goal_mocap_id_        = -1;
    int gait_param_id_        = -1;
    int gait_switch_param_id_ = -1;
    int flip_dir_param_id_    = -1;
    int biped_type_param_id_  = -1;
    int cadence_param_id_     = -1;
    int amplitude_param_id_   = -1;
    int duty_param_id_        = -1;
    int arm_posture_param_id_ = -1;
    int upright_cost_id_      = -1;
    int balance_cost_id_      = -1;
    int height_cost_id_       = -1;
    int grf_cost_id_          = -1;
    double grf_weight_default_ = 0.0;
    int foot_geom_id_[kNumFoot];
    int shoulder_body_id_[kNumFoot];
    int abduction_joint_id_[kNumFoot] = {-1, -1, -1, -1};
    int hip_joint_id_[kNumFoot] = {-1, -1, -1, -1};
    int knee_joint_id_[kNumFoot] = {-1, -1, -1, -1};
    int debug_grf_param_id_   = -1;
    int hind_grf_align_sensor_id_ = -1;

    // Contact/target smoothing state for GRF alignment.
    mutable int contact_streak_[kNumFoot] = {0, 0, 0, 0};
    mutable double filtered_target_proj_[kNumFoot][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0}};
    mutable double last_filter_time_ = -1.0;

    // derived kinematic quantities describing flip trajectory
    double gravity_           = 0;
    double jump_vel_          = 0;
    double flight_time_       = 0;
    double jump_acc_          = 0;
    double crouch_time_       = 0;
    double leap_time_         = 0;
    double jump_time_         = 0;
    double crouch_vel_        = 0;
    double land_time_         = 0;
    double land_acc_          = 0;
    double flight_rot_vel_    = 0;
    double jump_rot_vel_      = 0;
    double jump_rot_acc_      = 0;
    double land_rot_acc_      = 0;
    double total_mass_        = 0;

    // debug logging state shared across residual copies
    struct DebugLogState {
      std::mutex state_mutex;
      bool header_printed = false;
      int print_count = 0;
      double last_print_time = -std::numeric_limits<double>::infinity();
      double next_print_time = -std::numeric_limits<double>::infinity();
    };

    // CSV logging is handled by the shared `mjpc::CsvLogger` utility. The
    // previous per-task CsvLogState has been removed in favor of the single
    // shared logger to avoid truncation races and lock-order complexity.

    void MaybeLogStep(const mjModel* model, const mjData* data,
                      const FootContactInfo* contact_info,
                      const double* net_grf,
                      bool measurement_active) const;

    // Testing helpers (exposed for unit tests)
   public:
    struct MotorSelectionTestResult {
      bool valid = false;
      int index = -1;
      double angle = 0.0;
      double motor_proj[3] = {0.0, 0.0, 0.0};
      double normal_proj[3] = {0.0, 0.0, 0.0};
    };
    static MotorSelectionTestResult SelectMotorUsingNormalForTest(
        const double contact_normal[3], const double motor_vectors[2][3],
        const double plane_normal[3], double blend_beta = 0.0);
    static bool ComputePlaneFromPointsForTest(const double hip_anchor[3],
                                             const double knee_anchor[3],
                                             const double foot_point[3],
                                             double plane_normal_out[3]);
    // Compute per-foot residual (intended for unit tests). Returns true and
    // writes `out_res` if a meaningful residual could be computed (e.g., for
    // a hind foot with stable contact); otherwise returns false.
    static bool ComputePerFootResidualForTest(const double foot_force[3],
                          const double contact_normal[3],
                          const double hip_anchor[3],
                          const double knee_anchor[3],
                          const double foot_point[3],
                          double grf_weight,
                          double out_res[3]);

    mutable std::shared_ptr<DebugLogState> debug_log_state_;
  };

  QuadrupedFlatMod() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

  // call base-class Reset, save task-related ids
  void ResetLocked(const mjModel* model) override;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;

  // true when we want to suppress controller torques during startup settling
  bool ShouldHoldStartup(double time) const;

  // Note: startup hold is handled centrally in `app.cc` to behave the same
  // for both vanilla and modified quadruped tasks.

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  friend class ResidualFn;
  ResidualFn residual_;
};

class QuadrupedHillMod : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const QuadrupedHillMod* task, int current_mode = 0)
      : mjpc::BaseResidualFn(task), current_mode_(current_mode) {}

    // --------------------- Residuals for quadruped task --------------------
    //   Number of residuals: 4
    //     Residual (0): position_z - average(foot position)_z - height_goal
    //     Residual (1): position - goal_position
    //     Residual (2): orientation - goal_orientation
    //     Residual (3): control
    //   Number of parameters: 1
    //     Parameter (1): height_goal
    // -----------------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  private:
   friend class QuadrupedHillMod;
    int current_mode_;
  };
  QuadrupedHillMod() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_MOD_QUADRUPED_H_
