#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PROJECT_NAME="phenix"
IMAGE_NAME="${PROJECT_NAME}-img"

BASE_IMAGE="ubuntu:22.04"
CASADI_REF="main"
ROS_DISTRO="humble"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -nc|--nvidiac)
      BASE_IMAGE="nvidia/cuda:13.0.1-runtime-ubuntu22.04"
      shift
      ;;
    -ng|--nvidiag)
      BASE_IMAGE="nvidia/opengl:1.2-glvnd-devel-ubuntu22.04"
      shift
      ;;
    --cuda-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    --casadi-ref)
      CASADI_REF="$2"
      shift 2
      ;;
    --ros-distro)
      ROS_DISTRO="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Использование: build_docker.sh [опции]
  -n, --nvidia           использовать базовый образ NVIDIA CUDA
      --cuda-image IMG   указать собственный базовый образ (заменяет -n)
      --casadi-ref REF   выбирать ветку/тег CasADi (по умолчанию main)
      --ros-distro NAME  дистрибутив ROS 2 (по умолчанию humble)
EOF
      exit 0
      ;;
    *)
      echo "Неизвестный ключ: $1" >&2
      exit 1
      ;;
  esac
done

echo "[+] Сборка ${IMAGE_NAME}"
echo "    BASE_IMAGE=${BASE_IMAGE}"
echo "    CASADI_REF=${CASADI_REF}"
echo "    ROS_DISTRO=${ROS_DISTRO}"

docker build \
  --network=host \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg CASADI_REF="${CASADI_REF}" \
  --build-arg ROS_DISTRO="${ROS_DISTRO}" \
  -t "${IMAGE_NAME}" \
  -f "${ROOT_DIR}/docker/Dockerfile" \
  "${ROOT_DIR}"
