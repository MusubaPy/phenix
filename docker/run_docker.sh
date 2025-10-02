#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PROJECT_NAME="phenix"
IMAGE_NAME="${PROJECT_NAME}-img"

usage() {
  cat <<'EOF'
Использование: run_docker.sh [опция]
  без флагов             X11 + Mesa
  -n  --nvidia           X11 + NVIDIA GPU
  -w  --wayland          Wayland + Mesa
  -wn --wayland-nvidia   Wayland + NVIDIA GPU
  -h  --help             показать эту справку
EOF
}

MODE="x11-mesa"
if [[ $# -gt 0 ]]; then
  case "$1" in
    -n|--nvidia) MODE="x11-nvidia" ;;
    -w|--wayland) MODE="wayland-mesa" ;;
    -wn|--wayland-nvidia) MODE="wayland-nvidia" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Неизвестный ключ: $1" >&2; usage; exit 1 ;;
  esac
fi

declare -a RUN_ARGS
RUN_ARGS=(
  "--rm" "-it"
  "--name" "${PROJECT_NAME}"
  "--net=host"
  "--privileged"
  "--ipc=host"
  "--security-opt" "seccomp=unconfined"
  "-v" "${ROOT_DIR}/${PROJECT_NAME}_ws:/${PROJECT_NAME}_ws"
  "-v" "$HOME/.ssh:/root/.ssh:ro"
  "-e" "GIT_SSH_COMMAND=ssh -o StrictHostKeyChecking=no"
)

docker_supports_runtime() {
  local runtime_name="$1"
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi

  local runtimes
  if ! runtimes=$(docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null); then
    return 1
  fi

  grep -qE "(^| )${runtime_name}( |$)" <<<"${runtimes}"
}

is_wsl_environment() {
  if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    return 0
  fi
  if [[ -f /proc/sys/kernel/osrelease ]] && grep -qi "microsoft" /proc/sys/kernel/osrelease 2>/dev/null; then
    return 0
  fi
  return 1
}

ensure_x_access() {
  if is_wsl_environment; then
    return
  fi
  if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker >/dev/null 2>&1 || true
  else
    echo "[!] xhost не найден, приложения X11 могут не стартовать"
  fi
}

maybe_mount_xauth() {
  local auth_path="${XAUTHORITY:-}"
  if [[ -z "$auth_path" ]]; then
    auth_path="$HOME/.Xauthority"
  fi

  if [[ -f "$auth_path" ]]; then
    RUN_ARGS+=("-e" "XAUTHORITY=${auth_path}")
    RUN_ARGS+=("-v" "${auth_path}:${auth_path}:ro")
  else
    echo "[!] Файл Xauthority не найден, используется xhost"
  fi
}

wayland_mount() {
  if [[ "${IS_WSL}" -eq 1 ]]; then
    local wl_display="${WAYLAND_DISPLAY:-wayland-0}"
    local runtime="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
    local pulse_server="${PULSE_SERVER:-unix:/mnt/wslg/PulseServer}"
    RUN_ARGS+=("-e" "WAYLAND_DISPLAY=${wl_display}")
    RUN_ARGS+=("-e" "XDG_RUNTIME_DIR=${runtime}")
    RUN_ARGS+=("-e" "PULSE_SERVER=${pulse_server}")
    RUN_ARGS+=("-v" "/mnt/wslg:/mnt/wslg")
    RUN_ARGS+=("-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw")
  else
    local runtime="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    local wl_display="${WAYLAND_DISPLAY:-wayland-0}"
    RUN_ARGS+=("-e" "WAYLAND_DISPLAY=${wl_display}")
    RUN_ARGS+=("-e" "XDG_RUNTIME_DIR=/tmp")
    RUN_ARGS+=("-e" "PULSE_RUNTIME_PATH=/tmp/pulse")
    if [[ -S "${runtime}/${wl_display}" ]]; then
      RUN_ARGS+=("-v" "${runtime}/${wl_display}:/tmp/${wl_display}")
    else
      echo "[!] Wayland сокет ${runtime}/${wl_display} не найден"
    fi
    if [[ -d "${runtime}/pulse" ]]; then
      RUN_ARGS+=("-v" "${runtime}/pulse:/tmp/pulse")
    else
      echo "[!] PulseAudio runtime ${runtime}/pulse не найден"
    fi
    RUN_ARGS+=("-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw")
  fi
}

wsl_nvidia_mounts() {
  if [[ "${IS_WSL}" -ne 1 ]]; then
    return
  fi

  local wsl_lib="/usr/lib/wsl/lib"
  if [[ -d "${wsl_lib}" ]]; then
    RUN_ARGS+=("-v" "${wsl_lib}:${wsl_lib}:ro")
    RUN_ARGS+=("-e" "WSL_GPU_LIB_DIR=${wsl_lib}")
  else
    echo "[!] Не найден каталог ${wsl_lib}, драйвер NVIDIA WSL может быть недоступен" >&2
  fi

  if [[ -e /dev/dxg ]]; then
    RUN_ARGS+=("--device" "/dev/dxg")
  fi

  RUN_ARGS+=("-e" "__NV_PRIME_RENDER_OFFLOAD=1")
  RUN_ARGS+=("-e" "__NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0")
  RUN_ARGS+=("-e" "__VK_LAYER_NV_optimus=1")
}

IS_WSL=0
if is_wsl_environment; then
  IS_WSL=1
fi

case "$MODE" in
  x11-mesa)
    ensure_x_access
    maybe_mount_xauth
    RUN_ARGS+=(
      "-e" "DISPLAY"
      "-e" "QT_X11_NO_MITSHM=1"
      "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    )
    ;;
  x11-nvidia)
    ensure_x_access
    maybe_mount_xauth
    RUN_ARGS+=(
      "--gpus" "all"
      "-e" "DISPLAY"
      "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
      "-e" "QT_X11_NO_MITSHM=1"
      "-e" "NVIDIA_VISIBLE_DEVICES=all"
      "-e" "NVIDIA_DRIVER_CAPABILITIES=all"
      "-e" "__GLX_VENDOR_LIBRARY_NAME=nvidia"
    )
    wsl_nvidia_mounts
    if [[ "${IS_WSL}" -eq 0 ]]; then
      if docker_supports_runtime "nvidia"; then
        RUN_ARGS+=("--runtime" "nvidia")
      else
        cat <<'EOF' >&2
[x] Docker runtime 'nvidia' не найден.
    Убедитесь, что установлен NVIDIA Container Toolkit и выполнено:
      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker
EOF
        exit 1
      fi
    fi
    ;;
  wayland-mesa)
    wayland_mount
    ;;
  wayland-nvidia)
    RUN_ARGS+=("--gpus" "all")
    wsl_nvidia_mounts
    if [[ "${IS_WSL}" -eq 0 ]]; then
      if docker_supports_runtime "nvidia"; then
        RUN_ARGS+=("--runtime" "nvidia")
      else
        cat <<'EOF' >&2
[x] Docker runtime 'nvidia' не найден.
    Убедитесь, что установлен NVIDIA Container Toolkit и выполнено:
      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker
EOF
        exit 1
      fi
    fi
    RUN_ARGS+=(
      "-e" "NVIDIA_VISIBLE_DEVICES=all"
      "-e" "NVIDIA_DRIVER_CAPABILITIES=graphics,display,video,compute,utility"
      "-e" "__GLX_VENDOR_LIBRARY_NAME=nvidia"
      "-e" "__NV_PRIME_RENDER_OFFLOAD=1"
      "-e" "__NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0"
      "-e" "__VK_LAYER_NV_optimus=1"
    )
    wayland_mount
    ;;
esac

# Устройства по умолчанию (позволяют использовать GPU/джойстики и т.п.)
RUN_ARGS+=("-v" "/dev:/dev")

echo "[+] Запуск контейнера ${IMAGE_NAME} (режим: ${MODE})"
docker run "${RUN_ARGS[@]}" "${IMAGE_NAME}"

if [[ -d "${ROOT_DIR}/.git" ]]; then
  (
    cd "${ROOT_DIR}/.git" >/dev/null
    sudo chgrp -R "$(id -g -n "$(whoami)")" . || true
    sudo chmod -R g+rwX . || true
    sudo find . -type d -exec chmod g+s '{}' + || true
  )
fi
