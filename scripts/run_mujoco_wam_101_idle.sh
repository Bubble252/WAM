#!/usr/bin/env bash
#
# Run a MuJoCo-WAM generation job on 101 without taking a busy GPU.
#
# The script is intended to be copied with the repository to:
#   /vepfs-mlp2/c20250405/400040/transfer/WAM/repo/WAM
#
# It waits for a GPU with no compute process, <=2 GiB allocated memory, and
# <=10% utilization.  It never kills or preempts an existing process.

set -euo pipefail

WAM_REMOTE_ROOT="${WAM_REMOTE_ROOT:-/vepfs-mlp2/c20250405/400040/transfer/WAM}"
WAM_REPO_ROOT="${WAM_REPO_ROOT:-${WAM_REMOTE_ROOT}/repo/WAM}"
WAM_DATA_ROOT="${WAM_DATA_ROOT:-${WAM_REMOTE_ROOT}/data/mujoco_wam_v0}"
WAM_ENV_ROOT="${WAM_ENV_ROOT:-${WAM_REMOTE_ROOT}/envs/mujoco_wam_py311}"
WAM_PYTHON="${WAM_PYTHON:-${WAM_ENV_ROOT}/bin/python}"
WAM_POLL_SECONDS="${WAM_POLL_SECONDS:-60}"
WAM_GPU_MEMORY_LIMIT_MB="${WAM_GPU_MEMORY_LIMIT_MB:-2048}"
WAM_GPU_UTIL_LIMIT="${WAM_GPU_UTIL_LIMIT:-10}"
WAM_TRAJECTORIES_PER_FAMILY="${WAM_TRAJECTORIES_PER_FAMILY:-300}"
WAM_MASTER_SEED="${WAM_MASTER_SEED:-20261007}"

JOB_ID="mujoco_wam_$(date -u +%Y%m%dT%H%M%SZ)_$$"
JOB_DIR="${WAM_REMOTE_ROOT}/jobs"
LOG_DIR="${WAM_REMOTE_ROOT}/logs"
mkdir -p "${JOB_DIR}" "${LOG_DIR}" "${WAM_DATA_ROOT}"
LOG_PATH="${LOG_DIR}/${JOB_ID}.log"
STATUS_PATH="${JOB_DIR}/${JOB_ID}.json"

exec > >(tee -a "${LOG_PATH}") 2>&1

status="starting"
gpu_index=""
gpu_uuid=""
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_status() {
  local exit_code="${1:-null}"
  cat > "${STATUS_PATH}" <<EOF
{
  "job_id": "${JOB_ID}",
  "host": "$(hostname)",
  "status": "${status}",
  "exit_code": ${exit_code},
  "started_at": "${started_at}",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "gpu_index": "${gpu_index}",
  "gpu_uuid": "${gpu_uuid}",
  "repo_root": "${WAM_REPO_ROOT}",
  "data_root": "${WAM_DATA_ROOT}",
  "log_path": "${LOG_PATH}",
  "python": "${WAM_PYTHON}",
  "trajectories_per_family": ${WAM_TRAJECTORIES_PER_FAMILY},
  "master_seed": ${WAM_MASTER_SEED}
}
EOF
}

on_exit() {
  local code=$?
  if [[ "${code}" -eq 0 ]]; then
    status="completed"
  else
    status="failed"
  fi
  write_status "${code}"
  echo "job_status=${status} exit_code=${code} status_file=${STATUS_PATH}"
}
trap on_exit EXIT

select_idle_gpu() {
  local row index uuid memory util compute_pids
  while IFS=',' read -r index uuid memory util; do
    index="${index//[[:space:]]/}"
    uuid="${uuid//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [[ -n "${index}" && "${memory}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || continue
    compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${index}" 2>/dev/null || true)"
    if [[ -z "$(printf '%s' "${compute_pids}" | tr -d '[:space:]')" ]] \
      && (( memory <= WAM_GPU_MEMORY_LIMIT_MB )) \
      && (( util <= WAM_GPU_UTIL_LIMIT )); then
      printf '%s|%s|%s|%s\n' "${index}" "${uuid}" "${memory}" "${util}"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  return 1
}

echo "host=$(hostname)"
echo "repo_root=${WAM_REPO_ROOT}"
echo "data_root=${WAM_DATA_ROOT}"
echo "log_path=${LOG_PATH}"
echo "status_path=${STATUS_PATH}"
echo "waiting_for_idle_gpu=true"
status="waiting_for_idle_gpu"
write_status

while true; do
  candidate="$(select_idle_gpu || true)"
  if [[ -n "${candidate}" ]]; then
    IFS='|' read -r gpu_index gpu_uuid gpu_memory gpu_util <<< "${candidate}"
    sleep 5
    second_candidate="$(select_idle_gpu || true)"
    if [[ -n "${second_candidate}" ]]; then
      IFS='|' read -r second_index second_uuid second_memory second_util <<< "${second_candidate}"
      if [[ "${gpu_index}" == "${second_index}" && "${gpu_uuid}" == "${second_uuid}" ]]; then
        break
      fi
    fi
  fi
  echo "no stable idle GPU; sleeping ${WAM_POLL_SECONDS}s"
  sleep "${WAM_POLL_SECONDS}"
done

status="running"
write_status
echo "selected_gpu_index=${gpu_index}"
echo "selected_gpu_uuid=${gpu_uuid}"
echo "selected_gpu_memory_mb=${gpu_memory}"
echo "selected_gpu_utilization=${gpu_util}"

export CUDA_VISIBLE_DEVICES="${gpu_index}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

generator_args=(
  --output-root "${WAM_DATA_ROOT}"
  --trajectories-per-family "${WAM_TRAJECTORIES_PER_FAMILY}"
  --master-seed "${WAM_MASTER_SEED}"
)
if [[ "${WAM_NO_VIDEO:-0}" == "1" ]]; then
  generator_args+=(--no-video)
fi

"${WAM_PYTHON}" "${WAM_REPO_ROOT}/scripts/generate_mujoco_wam.py" "${generator_args[@]}" "$@"
