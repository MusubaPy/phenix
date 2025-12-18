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

#include "mjpc/tasks/tasks.h"

#include <memory>
#include <vector>

#include "mjpc/task.h"

// Optional task headers: include them only if available in the source tree.
#if __has_include("mjpc/tasks/acrobot/acrobot.h")
#include "mjpc/tasks/acrobot/acrobot.h"
#define MJPC_HAVE_ACROBOT 1
#endif
#if __has_include("mjpc/tasks/allegro/allegro.h")
#include "mjpc/tasks/allegro/allegro.h"
#define MJPC_HAVE_ALLEGRO 1
#endif
#if __has_include("mjpc/tasks/bimanual/handover/handover.h")
#include "mjpc/tasks/bimanual/handover/handover.h"
#define MJPC_HAVE_BIMANUAL_HANDOVER 1
#endif
#if __has_include("mjpc/tasks/bimanual/insert/insert.h")
#include "mjpc/tasks/bimanual/insert/insert.h"
#define MJPC_HAVE_BIMANUAL_INSERT 1
#endif
#if __has_include("mjpc/tasks/bimanual/reorient/reorient.h")
#include "mjpc/tasks/bimanual/reorient/reorient.h"
#define MJPC_HAVE_BIMANUAL_REORIENT 1
#endif
#if __has_include("mjpc/tasks/cartpole/cartpole.h")
#include "mjpc/tasks/cartpole/cartpole.h"
#define MJPC_HAVE_CARTPOLE 1
#endif
#if __has_include("mjpc/tasks/fingers/fingers.h")
#include "mjpc/tasks/fingers/fingers.h"
#define MJPC_HAVE_FINGERS 1
#endif
#if __has_include("mjpc/tasks/humanoid/interact/interact.h")
#include "mjpc/tasks/humanoid/interact/interact.h"
#define MJPC_HAVE_HUMANOID_INTERACT 1
#endif
#if __has_include("mjpc/tasks/humanoid/stand/stand.h")
#include "mjpc/tasks/humanoid/stand/stand.h"
#define MJPC_HAVE_HUMANOID_STAND 1
#endif
#if __has_include("mjpc/tasks/humanoid/tracking/tracking.h")
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#define MJPC_HAVE_HUMANOID_TRACKING 1
#endif
#if __has_include("mjpc/tasks/humanoid/walk/walk.h")
#include "mjpc/tasks/humanoid/walk/walk.h"
#define MJPC_HAVE_HUMANOID_WALK 1
#endif
#if __has_include("mjpc/tasks/manipulation/manipulation.h")
#include "mjpc/tasks/manipulation/manipulation.h"
#define MJPC_HAVE_MANIPULATION 1
#endif
// DEEPMIND INTERNAL IMPORT
#if __has_include("mjpc/tasks/op3/stand.h")
#include "mjpc/tasks/op3/stand.h"
#define MJPC_HAVE_OP3 1
#endif
#if __has_include("mjpc/tasks/panda/panda.h")
#include "mjpc/tasks/panda/panda.h"
#define MJPC_HAVE_PANDA 1
#endif
#if __has_include("mjpc/tasks/particle/particle.h")
#include "mjpc/tasks/particle/particle.h"
#define MJPC_HAVE_PARTICLE 1
#endif
#if __has_include("mjpc/tasks/quadrotor/quadrotor.h")
#include "mjpc/tasks/quadrotor/quadrotor.h"
#define MJPC_HAVE_QUADROTOR 1
#endif
#if __has_include("mjpc/tasks/quadruped_vanila/quadruped_vanila.h")
#include "mjpc/tasks/quadruped_vanila/quadruped_vanila.h"
#define MJPC_HAVE_QUADRUPED_VANILA 1
#endif
#if __has_include("mjpc/tasks/quadruped_mod/quadruped_mod.h")
#include "mjpc/tasks/quadruped_mod/quadruped_mod.h"
#define MJPC_HAVE_QUADRUPED_MOD 1
#endif
#if __has_include("mjpc/tasks/rubik/solve.h")
#include "mjpc/tasks/rubik/solve.h"
#define MJPC_HAVE_RUBIK 1
#endif
#if __has_include("mjpc/tasks/shadow_reorient/hand.h")
#include "mjpc/tasks/shadow_reorient/hand.h"
#define MJPC_HAVE_SHADOW_REORIENT 1
#endif
#if __has_include("mjpc/tasks/swimmer/swimmer.h")
#include "mjpc/tasks/swimmer/swimmer.h"
#define MJPC_HAVE_SWIMMER 1
#endif
#if __has_include("mjpc/tasks/walker/walker.h")
#include "mjpc/tasks/walker/walker.h"
#define MJPC_HAVE_WALKER 1
#endif

namespace mjpc {

std::vector<std::shared_ptr<Task>> GetTasks() {
  std::vector<std::shared_ptr<Task>> tasks;
#ifdef MJPC_HAVE_ACROBOT
  tasks.push_back(std::make_shared<Acrobot>());
#endif
#ifdef MJPC_HAVE_ALLEGRO
  tasks.push_back(std::make_shared<Allegro>());
#endif
#ifdef MJPC_HAVE_BIMANUAL_HANDOVER
  tasks.push_back(std::make_shared<aloha::Handover>());
#endif
#ifdef MJPC_HAVE_BIMANUAL_INSERT
  tasks.push_back(std::make_shared<aloha::Insert>());
#endif
#ifdef MJPC_HAVE_BIMANUAL_REORIENT
  tasks.push_back(std::make_shared<aloha::Reorient>());
#endif
#ifdef MJPC_HAVE_CARTPOLE
  tasks.push_back(std::make_shared<Cartpole>());
#endif
#ifdef MJPC_HAVE_FINGERS
  tasks.push_back(std::make_shared<Fingers>());
#endif
#ifdef MJPC_HAVE_HUMANOID_INTERACT
  tasks.push_back(std::make_shared<humanoid::Interact>());
#endif
#ifdef MJPC_HAVE_HUMANOID_STAND
  tasks.push_back(std::make_shared<humanoid::Stand>());
#endif
#ifdef MJPC_HAVE_HUMANOID_TRACKING
  tasks.push_back(std::make_shared<humanoid::Tracking>());
#endif
#ifdef MJPC_HAVE_HUMANOID_WALK
  tasks.push_back(std::make_shared<humanoid::Walk>());
#endif
#ifdef MJPC_HAVE_MANIPULATION
  tasks.push_back(std::make_shared<manipulation::Bring>());
#endif
#ifdef MJPC_HAVE_OP3
  tasks.push_back(std::make_shared<OP3>());
#endif
#ifdef MJPC_HAVE_PANDA
  tasks.push_back(std::make_shared<Panda>());
#endif
#ifdef MJPC_HAVE_PARTICLE
  tasks.push_back(std::make_shared<Particle>());
#endif
  // ParticleFixed is a helper inside particle; include only if particle exists.
#ifdef MJPC_HAVE_PARTICLE
  tasks.push_back(std::make_shared<ParticleFixed>());
#endif
#ifdef MJPC_HAVE_RUBIK
  tasks.push_back(std::make_shared<Rubik>());
#endif
#ifdef MJPC_HAVE_SHADOW_REORIENT
  tasks.push_back(std::make_shared<ShadowReorient>());
#endif
#ifdef MJPC_HAVE_QUADROTOR
  tasks.push_back(std::make_shared<Quadrotor>());
#endif
#ifdef MJPC_HAVE_QUADRUPED_VANILA
  tasks.push_back(std::make_shared<QuadrupedFlat>());
  tasks.push_back(std::make_shared<QuadrupedHill>());
#endif
#ifdef MJPC_HAVE_QUADRUPED_MOD
  tasks.push_back(std::make_shared<QuadrupedFlatMod>());
  tasks.push_back(std::make_shared<QuadrupedHillMod>());
#endif
#ifdef MJPC_HAVE_SWIMMER
  tasks.push_back(std::make_shared<Swimmer>());
#endif
#ifdef MJPC_HAVE_WALKER
  tasks.push_back(std::make_shared<Walker>());
#endif

  return tasks;
}
}  // namespace mjpc
