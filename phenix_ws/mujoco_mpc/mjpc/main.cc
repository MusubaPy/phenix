// Copyright 2021 DeepMind Technologies Limited
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

#include <iostream>
#include <string>
#include <vector>
#include <absl/flags/parse.h>

#include <absl/flags/flag.h>
#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "mjpc/app.h"
#include "mjpc/tasks/tasks.h"
// shared utilities
#include "mjpc/common/csv_logger.h"

ABSL_FLAG(std::string, task, "Quadruped Flat",
          "Which model to load on startup.");

ABSL_FLAG(double, max_sim_time, 60.0,
          "Maximum simulation time in seconds before mjpc exits.");

ABSL_FLAG(double, internal_grf_align_weight, 1e-3,
          "Internal Kuznetsov GRF alignment residual weight (per-foot scale). Set to 0 to disable.");

// machinery for replacing command line error by a macOS dialog box
// when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default")))
extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char** argv) {
  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif
  absl::ParseCommandLine(argc, argv);

  // Allow overriding internal GRF align weight via environment variable
  if (const char* env = std::getenv("MJPC_INTERNAL_GRF_ALIGN_WEIGHT")) {
    try {
      double v = std::stod(std::string(env));
      absl::SetFlag(&FLAGS_internal_grf_align_weight, v);
    } catch (...) {
      // ignore parse errors
    }
  }

  std::string task_name = absl::GetFlag(FLAGS_task);

  // If invoked as `mjpc_mod` and the user did not override --task (left as the
  // default "Quadruped Flat"), prefer the mod task variant by default so the
  // binary behaves as expected when run without arguments.
  std::string progname = argv[0] ? std::string(argv[0]) : std::string();
  if (absl::StrContains(progname, "mjpc_mod") &&
      absl::EqualsIgnoreCase(task_name, "Quadruped Flat")) {
    task_name = "Quadruped Flat (mod)";
  }
  auto tasks = mjpc::GetTasks();
  int task_id = -1;
  for (int i = 0; i < tasks.size(); i++) {
    if (absl::EqualsIgnoreCase(task_name, tasks[i]->Name())) {
      task_id = i;
      break;
    }
  }
  if (task_id == -1) {
    std::cerr << "Invalid --task flag: '" << task_name
              << "'. Valid values:\n";
    for (int i = 0; i < tasks.size(); i++) {
      std::cerr << "  " << tasks[i]->Name() << "\n";
    }
    mju_error("Invalid --task flag.");
  }

  // Initialize shared CSV logger early so the file (and directories) are
  // created immediately and reported to the user as an absolute path. We
  // pass the default path used elsewhere; the logger will prefer the
  // MJPC_CSV_LOG env var if it is set.
  try {
    mjpc::CsvLogger::Instance().Init("logs/quadruped_log.csv");
  } catch (...) {
    // best-effort; don't fail startup on logger issues
  }

  mjpc::StartApp(tasks, task_id);  // start with quadruped flat
  return 0;
}
