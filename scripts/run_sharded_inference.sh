#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_sharded_inference.sh <30s|60s> [options]

Options:
  --prompts PATH            Prompt txt file. Default: prompts/MovieGenVideoBench_num128.txt
  --gpus IDS                Comma-separated GPU ids. Default: CUDA_VISIBLE_DEVICES or all GPUs from nvidia-smi
  --output-dir PATH         Output directory. Default: longlive-30s or longlive-60s
  --config PATH             Config template. Default: configs/longlive_inference_30s.yaml or 60s
  --master-port-base PORT   Base port for shard torchrun processes. Default: 29500
  --force                   Remove an existing non-empty output directory before running
  --dry-run                 Print shard allocation and exit without launching inference
  -h, --help                Show this help
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

abspath() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$path"
  fi
}

make_shard_config() {
  local template_path="$1"
  local output_path="$2"
  local prompt_path="$3"
  local shard_output_dir="$4"

  "$PYTHON_BIN" - "$template_path" "$output_path" "$prompt_path" "$shard_output_dir" <<'PY'
import json
import re
import sys
from pathlib import Path

template_path, output_path, prompt_path, shard_output_dir = sys.argv[1:5]
text = Path(template_path).read_text(encoding="utf-8")

replacements = {
    "data_path": prompt_path,
    "output_folder": shard_output_dir,
    "save_with_index": True,
    "num_samples": 1,
}

for key, value in replacements.items():
    pattern = rf"(?m)^{re.escape(key)}:\s*.*$"
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = json.dumps(value)
    replacement = f"{key}: {rendered}"
    if re.search(pattern, text):
        text = re.sub(pattern, replacement, text, count=1)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += replacement + "\n"

Path(output_path).write_text(text, encoding="utf-8")
PY
}

write_prompts_csv() {
  local prompt_path="$1"
  local csv_path="$2"

  "$PYTHON_BIN" - "$prompt_path" "$csv_path" <<'PY'
import csv
import sys
from pathlib import Path

prompt_path, csv_path = sys.argv[1:3]
lines = Path(prompt_path).read_text(encoding="utf-8").splitlines()

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "prompt"])
    for idx, prompt in enumerate(lines):
        writer.writerow([idx, prompt])
PY
}

DURATION=""
PROMPT_FILE="prompts/MovieGenVideoBench_num128.txt"
GPU_SPEC=""
OUTPUT_DIR=""
CONFIG_TEMPLATE=""
MASTER_PORT_BASE=29500
FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    30s|60s)
      [[ -z "$DURATION" ]] || die "Duration already set to $DURATION"
      DURATION="$1"
      shift
      ;;
    --prompts)
      [[ $# -ge 2 ]] || die "--prompts requires a path"
      PROMPT_FILE="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus requires a comma-separated list"
      GPU_SPEC="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a path"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || die "--config requires a path"
      CONFIG_TEMPLATE="$2"
      shift 2
      ;;
    --master-port-base)
      [[ $# -ge 2 ]] || die "--master-port-base requires a value"
      MASTER_PORT_BASE="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$DURATION" ]] || die "Duration is required: 30s or 60s"
[[ "$MASTER_PORT_BASE" =~ ^[0-9]+$ ]] || die "--master-port-base must be an integer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

require_cmd uv

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  die "Missing required command: python3 or python"
fi

if [[ -z "$CONFIG_TEMPLATE" ]]; then
  if [[ "$DURATION" == "30s" ]]; then
    CONFIG_TEMPLATE="configs/longlive_inference_30s.yaml"
  else
    CONFIG_TEMPLATE="configs/longlive_inference_60s.yaml"
  fi
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$DURATION" == "30s" ]]; then
    OUTPUT_DIR="longlive-30s"
  else
    OUTPUT_DIR="longlive-60s"
  fi
fi

PROMPT_FILE="$(abspath "$PROMPT_FILE")"
CONFIG_TEMPLATE="$(abspath "$CONFIG_TEMPLATE")"
OUTPUT_DIR="$(abspath "$OUTPUT_DIR")"

[[ -f "$PROMPT_FILE" ]] || die "Prompt file not found: $PROMPT_FILE"
[[ -f "$CONFIG_TEMPLATE" ]] || die "Config template not found: $CONFIG_TEMPLATE"

if [[ -n "$GPU_SPEC" ]]; then
  IFS=',' read -r -a RAW_GPUS <<< "$GPU_SPEC"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a RAW_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  require_cmd nvidia-smi
  mapfile -t RAW_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | sed 's/[[:space:]]//g')
fi

GPUS=()
for gpu in "${RAW_GPUS[@]:-}"; do
  gpu="${gpu//[[:space:]]/}"
  [[ -n "$gpu" ]] || continue
  GPUS+=("$gpu")
done

((${#GPUS[@]} > 0)) || die "No GPUs resolved. Pass --gpus or set CUDA_VISIBLE_DEVICES."

mapfile -t PROMPTS < "$PROMPT_FILE"
TOTAL_PROMPTS=${#PROMPTS[@]}
(($TOTAL_PROMPTS > 0)) || die "Prompt file is empty: $PROMPT_FILE"

GPU_COUNT=${#GPUS[@]}
BASE_COUNT=$(( TOTAL_PROMPTS / GPU_COUNT ))

declare -a SHARD_IDS=()
declare -a SHARD_GPUS=()
declare -a SHARD_STARTS=()
declare -a SHARD_COUNTS=()
declare -a SHARD_PROMPT_FILES=()
declare -a SHARD_CONFIGS=()
declare -a SHARD_OUTPUTS=()
declare -a SHARD_LOGS=()
declare -a SHARD_PORTS=()

current_start=0
for (( shard_id=0; shard_id<GPU_COUNT; shard_id++ )); do
  if (( shard_id == GPU_COUNT - 1 )); then
    shard_count=$(( TOTAL_PROMPTS - BASE_COUNT * (GPU_COUNT - 1) ))
  else
    shard_count=$BASE_COUNT
  fi

  if (( shard_count <= 0 )); then
    continue
  fi

  SHARD_IDS+=("$shard_id")
  SHARD_GPUS+=("${GPUS[$shard_id]}")
  SHARD_STARTS+=("$current_start")
  SHARD_COUNTS+=("$shard_count")
  SHARD_PORTS+=("$(( MASTER_PORT_BASE + shard_id ))")
  current_start=$(( current_start + shard_count ))
done

for i in "${!SHARD_IDS[@]}"; do
  printf '[PLAN] shard=%s gpu=%s start=%s count=%s port=%s\n' \
    "${SHARD_IDS[$i]}" "${SHARD_GPUS[$i]}" "${SHARD_STARTS[$i]}" "${SHARD_COUNTS[$i]}" "${SHARD_PORTS[$i]}"
done

if (( DRY_RUN == 1 )); then
  exit 0
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  if (( FORCE == 1 )); then
    rm -rf "$OUTPUT_DIR"
  else
    [[ -d "$OUTPUT_DIR" ]] || die "Output path exists and is not a directory: $OUTPUT_DIR"
    if find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
      die "Output directory already exists and is not empty: $OUTPUT_DIR (pass --force to overwrite)"
    fi
  fi
fi

LOG_DIR="$OUTPUT_DIR/logs"
STAGING_DIR="$OUTPUT_DIR/.staging"
PROMPT_SHARD_DIR="$STAGING_DIR/prompts"
CONFIG_SHARD_DIR="$STAGING_DIR/configs"
RAW_OUTPUT_DIR="$STAGING_DIR/raw"

mkdir -p "$LOG_DIR" "$PROMPT_SHARD_DIR" "$CONFIG_SHARD_DIR" "$RAW_OUTPUT_DIR"

RUN_SUCCEEDED=0
cleanup() {
  if (( RUN_SUCCEEDED == 1 )); then
    rm -rf "$STAGING_DIR"
  else
    echo "Preserving staging files for debugging: $STAGING_DIR" >&2
  fi
}
trap cleanup EXIT

write_prompts_csv "$PROMPT_FILE" "$OUTPUT_DIR/prompts.csv"

for i in "${!SHARD_IDS[@]}"; do
  shard_id="${SHARD_IDS[$i]}"
  start_idx="${SHARD_STARTS[$i]}"
  shard_count="${SHARD_COUNTS[$i]}"
  shard_gpu="${SHARD_GPUS[$i]}"
  shard_port="${SHARD_PORTS[$i]}"

  shard_prompt_file="$PROMPT_SHARD_DIR/shard_$(printf '%02d' "$shard_id").txt"
  shard_config_file="$CONFIG_SHARD_DIR/shard_$(printf '%02d' "$shard_id").yaml"
  shard_output_dir="$RAW_OUTPUT_DIR/shard_$(printf '%02d' "$shard_id")"
  shard_log_file="$LOG_DIR/shard_$(printf '%02d' "$shard_id")_gpu${shard_gpu}.log"

  mkdir -p "$shard_output_dir"
  : > "$shard_prompt_file"
  for (( prompt_idx=start_idx; prompt_idx<start_idx+shard_count; prompt_idx++ )); do
    printf '%s\n' "${PROMPTS[$prompt_idx]}" >> "$shard_prompt_file"
  done

  make_shard_config "$CONFIG_TEMPLATE" "$shard_config_file" "$shard_prompt_file" "$shard_output_dir"

  SHARD_PROMPT_FILES+=("$shard_prompt_file")
  SHARD_CONFIGS+=("$shard_config_file")
  SHARD_OUTPUTS+=("$shard_output_dir")
  SHARD_LOGS+=("$shard_log_file")
done

declare -a PIDS=()
for i in "${!SHARD_IDS[@]}"; do
  shard_id="${SHARD_IDS[$i]}"
  shard_gpu="${SHARD_GPUS[$i]}"
  shard_port="${SHARD_PORTS[$i]}"
  shard_config_file="${SHARD_CONFIGS[$i]}"
  shard_log_file="${SHARD_LOGS[$i]}"

  (
    set -euo pipefail
    echo "[INFO] shard=${shard_id} gpu=${shard_gpu} port=${shard_port} config=${shard_config_file}"
    CUDA_VISIBLE_DEVICES="$shard_gpu" uv run torchrun \
      --nproc_per_node=1 \
      --master_port="$shard_port" \
      inference.py \
      --config_path "$shard_config_file"
  ) >"$shard_log_file" 2>&1 &

  PIDS+=("$!")
done

remaining=${#PIDS[@]}
while (( remaining > 0 )); do
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]:-}"
    [[ -n "$pid" ]] || continue

    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        :
      else
        status=$?
        echo "Shard ${SHARD_IDS[$i]} failed on GPU ${SHARD_GPUS[$i]} with exit code ${status}. See ${SHARD_LOGS[$i]}" >&2
        for j in "${!PIDS[@]}"; do
          other_pid="${PIDS[$j]:-}"
          [[ -n "$other_pid" ]] || continue
          if [[ "$other_pid" != "$pid" ]]; then
            kill "$other_pid" 2>/dev/null || true
          fi
        done
        for j in "${!PIDS[@]}"; do
          other_pid="${PIDS[$j]:-}"
          [[ -n "$other_pid" ]] || continue
          wait "$other_pid" || true
        done
        exit "$status"
      fi

      PIDS[$i]=""
      remaining=$(( remaining - 1 ))
    fi
  done
  sleep 2
done

for i in "${!SHARD_IDS[@]}"; do
  start_idx="${SHARD_STARTS[$i]}"
  shard_count="${SHARD_COUNTS[$i]}"
  shard_output_dir="${SHARD_OUTPUTS[$i]}"

  for (( local_idx=0; local_idx<shard_count; local_idx++ )); do
    matches=( "$shard_output_dir"/rank0-"$local_idx"-0_*.mp4 )
    if (( ${#matches[@]} != 1 )); then
      die "Expected exactly one mp4 for shard ${SHARD_IDS[$i]} local index ${local_idx}, found ${#matches[@]} in $shard_output_dir"
    fi

    global_idx=$(( start_idx + local_idx ))
    dest_file="$(printf '%s/video_%03d.mp4' "$OUTPUT_DIR" "$global_idx")"
    mv "${matches[0]}" "$dest_file"
  done
done

final_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'video_*.mp4' | wc -l | tr -d ' ')
[[ "$final_count" == "$TOTAL_PROMPTS" ]] || die "Expected $TOTAL_PROMPTS videos, found $final_count in $OUTPUT_DIR"

RUN_SUCCEEDED=1
echo "Completed ${DURATION} inference for ${TOTAL_PROMPTS} prompts."
echo "Outputs: $OUTPUT_DIR"
echo "CSV: $OUTPUT_DIR/prompts.csv"
