// Simple thread-safe CSV logger shared across tasks.
#ifndef MJPC_COMMON_CSV_LOGGER_H_
#define MJPC_COMMON_CSV_LOGGER_H_

#include <mutex>
#include <fstream>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>

namespace mjpc {

class CsvLogger {
 public:
  static CsvLogger& Instance();

  // Initialize with optional path (MJPC_CSV_LOG env var takes precedence if present).
  void Init(const std::string& default_path);

  // Ensure actuator joint ids are set from model (size=model->nu)
  void EnsureActuatorIds(const mjModel* model);

  // Ensure header is written once. Caller prepares full header line (no trailing newline required).
  void EnsureHeader(const std::string& header_line);

  // Append a row prefix (everything before energy values). The logger will append
  // the accumulated energy columns and a phase tag and write the line.
  // t is current time, dt is delta from last_time, measurement_active controls accumulation.
  void AppendRow(const std::string& row_prefix, double t, double step_abs, double step_signed, bool measurement_active, int phase_tag);

  // Return last time seen by the logger (guarded by mutex). Returns a very
  // large negative value if no time has been recorded yet.
  double GetLastTime();

 private:
  CsvLogger();
  ~CsvLogger();

  std::mutex mutex_;
  std::ofstream stream_;
  bool stream_ready_ = false;
  bool header_written_ = false;
  double last_time_ = -1e300;
  double energy_abs_ = 0.0;
  double energy_signed_ = 0.0;
  std::string path_ = "logs/quadruped_log.csv";
  std::vector<int> actuator_joint_ids_;
};

}  // namespace mjpc

#endif  // MJPC_COMMON_CSV_LOGGER_H_
