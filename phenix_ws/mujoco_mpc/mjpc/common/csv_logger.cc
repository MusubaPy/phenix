#include "mjpc/common/csv_logger.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>

namespace mjpc {

CsvLogger& CsvLogger::Instance() {
  static CsvLogger instance;
  return instance;
}

CsvLogger::CsvLogger() {}
CsvLogger::~CsvLogger() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (stream_.is_open()) stream_.close();
}

void CsvLogger::Init(const std::string& default_path) {
  std::lock_guard<std::mutex> guard(mutex_);
  const char* env_path = std::getenv("MJPC_CSV_LOG");
  if (env_path && *env_path) path_ = env_path;
  else if (!default_path.empty()) path_ = default_path;
  std::filesystem::path csv_path(path_);
  // Normalize to an absolute path so file creation and diagnostics are
  // unambiguous regardless of the current working directory.
  try {
    csv_path = std::filesystem::absolute(csv_path);
    path_ = csv_path.string();
  } catch (...) {
    // If absolute conversion fails for any reason, fall back to the given path.
  }
  if (csv_path.has_parent_path() && !csv_path.parent_path().empty()) {
    std::error_code ec;
    std::filesystem::create_directories(csv_path.parent_path(), ec);
  }
  stream_.open(csv_path, std::ios::out | std::ios::trunc);
  if (!stream_.is_open()) {
    stream_ready_ = false;
    std::cerr << "CsvLogger::Init(): failed to open '" << csv_path.string() << "' (cwd=" << std::filesystem::current_path().string() << ")\n";
    return;
  }
  stream_ready_ = true;
  // Informational: print the absolute path of the opened CSV so users know
  // exactly where logs are being written.
  std::cerr << "CsvLogger::Init(): opened '" << csv_path.string() << "'\n";
  // Write an initial comment line so the file is non-empty even if the
  // process is interrupted before the task-specific header is written.
  // This helps users detect that logging was intended and where the file
  // lives.
  {
    auto now = std::chrono::system_clock::now();
    std::time_t tnow = std::chrono::system_clock::to_time_t(now);
    stream_ << "# MJPC CSV file created: " << std::ctime(&tnow);
    // ctime appends a newline; ensure the buffer hits disk.
    stream_.flush();
  }
}

void CsvLogger::EnsureActuatorIds(const mjModel* model) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!model) return;
  if (static_cast<int>(actuator_joint_ids_.size()) == model->nu) return;
  actuator_joint_ids_.resize(model->nu, -1);
  for (int i = 0; i < model->nu; ++i) actuator_joint_ids_[i] = model->actuator_trnid[2 * i];
}

void CsvLogger::EnsureHeader(const std::string& header_line) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!stream_ready_) Init(path_);
  if (!stream_ready_) return;
  if (!header_written_) {
    stream_ << header_line << '\n';
    header_written_ = true;
    // Ensure the header is flushed to disk so it is visible even on early
    // termination (e.g., SIGINT). This avoids losing the header when the
    // process is interrupted shortly after it is written.
    stream_.flush();
    const char* dbg = std::getenv("MJPC_CSV_DEBUG");
    if (dbg && *dbg) std::cerr << "CsvLogger::EnsureHeader(): wrote header\n";
  }
}

void CsvLogger::AppendRow(const std::string& row_prefix, double t, double step_abs, double step_signed, bool measurement_active, int phase_tag) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!stream_ready_) Init(path_);
  if (!stream_ready_) return;
  double dt = 0.0;
  if (std::isfinite(last_time_)) {
    dt = t - last_time_;
    if (dt < 0.0) dt = 0.0;
  }
  if (t <= last_time_ + 1e-9) return;
  if (measurement_active && dt > 0.0) {
    energy_signed_ += step_signed;
    energy_abs_ += step_abs;
  }
  // write: <row_prefix>,energy_abs,energy_signed,phase_tag\n
  stream_ << row_prefix << ',' << std::fixed << std::setprecision(6) << energy_abs_ << ',' << energy_signed_ << ',' << phase_tag << '\n';
  // Flush after writing each row so that rows are not left in user-space
  // buffers if the process is interrupted (SIGINT/SIGTERM).
  stream_.flush();
  const char* dbg = std::getenv("MJPC_CSV_DEBUG");
  if (dbg && *dbg) std::cerr << "CsvLogger::AppendRow(): wrote t=" << t << " abs=" << energy_abs_ << " signed=" << energy_signed_ << "\n";
  last_time_ = t;
}

double CsvLogger::GetLastTime() {
  std::lock_guard<std::mutex> guard(mutex_);
  return last_time_;
}

}  // namespace mjpc
