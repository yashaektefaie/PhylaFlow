#!/usr/bin/env bash
set -u

RUN_NAME="realstream_leafcap120_12451_phy256_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_ds2eval_replaysubset_mrbayes20k_20260503"
WORKDIR="/home/unix/yektefai/PhylaFlow"
CONFIG_PATH="/ewsc/yektefai/30272299/launch_configs_ewsc/configs/local_realstream_348299_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_ds2eval_replaysubset_mrbayes20k_20260503.yaml"
PYTHON_BIN="/ewsc/yektefai/envs/envs/pgt/bin/python"
CODEX_BIN="${CODEX_BIN:-/home/unix/yektefai/tools/codex}"

RUN_LOG="/ewsc/yektefai/phylaflow/logs/full_sanity_fixedpair_20260401/${RUN_NAME}.log"
METRICS_PATH="/ewsc/yektefai/phylaflow/metrics/full_sanity_fixedpair_20260401/${RUN_NAME}_metrics.jsonl"
CHECKPOINT_DIR="/ewsc/yektefai/phylaflow/checkpoints/full_sanity_fixedpair_20260401/${RUN_NAME}"
MONITOR_DIR="/ewsc/yektefai/phylaflow/logs/full_sanity_fixedpair_20260401/monitors/${RUN_NAME}"
MONITOR_LOG="${MONITOR_DIR}/monitor.log"
STATE_FILE="${MONITOR_DIR}/state.env"
ENV_FILE="${MONITOR_DIR}/last_run_env.env"
LOCK_FILE="${MONITOR_DIR}/monitor.lock"

POLL_SECONDS="${POLL_SECONDS:-120}"
CODEX_AUTOFIX_ENABLED="${CODEX_AUTOFIX_ENABLED:-1}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-14400}"
CRASH_COOLDOWN_SECONDS="${CRASH_COOLDOWN_SECONDS:-1800}"
STALE_LOG_SECONDS="${STALE_LOG_SECONDS:-7200}"

mkdir -p "$MONITOR_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '[%s] monitor already running for %s\n' "$(date -Is)" "$RUN_NAME" >> "$MONITOR_LOG"
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$MONITOR_LOG" >/dev/null
}

load_state() {
  LAST_SEEN_PID=""
  LAST_CODEX_AT=0
  LAST_METRICS_SIZE=0
  LAST_METRICS_LINE_COUNT=0
  LAST_INCIDENT_KEY=""

  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  {
    printf 'LAST_SEEN_PID=%q\n' "${LAST_SEEN_PID:-}"
    printf 'LAST_CODEX_AT=%q\n' "${LAST_CODEX_AT:-0}"
    printf 'LAST_METRICS_SIZE=%q\n' "${LAST_METRICS_SIZE:-0}"
    printf 'LAST_METRICS_LINE_COUNT=%q\n' "${LAST_METRICS_LINE_COUNT:-0}"
    printf 'LAST_INCIDENT_KEY=%q\n' "${LAST_INCIDENT_KEY:-}"
  } > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

find_run_pid() {
  ps -eo pid=,comm=,args= |
    awk -v cfg="$CONFIG_PATH" '
      $2 ~ /python/ && index($0, "-m run.run") && index($0, cfg) { print $1 }
    ' |
    tail -n 1
}

record_live_env() {
  local pid="$1"
  [[ -r "/proc/${pid}/environ" ]] || return 0

  {
    tr '\0' '\n' < "/proc/${pid}/environ" |
      awk -F= '
        $1 == "CUDA_VISIBLE_DEVICES" ||
        $1 == "WANDB_MODE" ||
        $1 == "WANDB_DIR" ||
        $1 == "WANDB_PROJECT" ||
        $1 == "PYTHONPATH" {
          print
        }
      '
  } > "${ENV_FILE}.tmp" 2>/dev/null || true
  mv "${ENV_FILE}.tmp" "$ENV_FILE"
}

append_file_tail() {
  local label="$1"
  local path="$2"
  local lines="$3"
  local out="$4"

  {
    printf '\n===== %s: %s =====\n' "$label" "$path"
    if [[ -f "$path" ]]; then
      tail -n "$lines" "$path"
    else
      printf 'missing\n'
    fi
  } >> "$out" 2>&1
}

build_incident_bundle() {
  local reason="$1"
  local pid="${2:-}"
  local now
  now="$(date +%Y%m%d_%H%M%S)"

  local incident_dir="${MONITOR_DIR}/incident_${now}"
  mkdir -p "$incident_dir"

  {
    printf 'run_name=%s\n' "$RUN_NAME"
    printf 'reason=%s\n' "$reason"
    printf 'observed_pid=%s\n' "$pid"
    printf 'last_seen_pid=%s\n' "${LAST_SEEN_PID:-}"
    printf 'time=%s\n' "$(date -Is)"
    printf 'workdir=%s\n' "$WORKDIR"
    printf 'config_path=%s\n' "$CONFIG_PATH"
    printf 'run_log=%s\n' "$RUN_LOG"
    printf 'metrics_path=%s\n' "$METRICS_PATH"
    printf 'checkpoint_dir=%s\n' "$CHECKPOINT_DIR"
  } > "${incident_dir}/summary.txt"

  ps -eo pid,ppid,lstart,etime,stat,pcpu,pmem,args > "${incident_dir}/ps_snapshot.txt" 2>&1 || true
  ps -eo pid,ppid,lstart,etime,stat,pcpu,pmem,args | grep -F "$CONFIG_PATH" > "${incident_dir}/matching_processes.txt" 2>&1 || true
  git -C "$WORKDIR" status --short > "${incident_dir}/git_status.txt" 2>&1 || true
  find "$CHECKPOINT_DIR" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort > "${incident_dir}/checkpoint_files.txt" || true
  cp "$ENV_FILE" "${incident_dir}/last_run_env.env" 2>/dev/null || true

  append_file_tail "run log tail" "$RUN_LOG" 500 "${incident_dir}/run_log_tail.txt"
  append_file_tail "metrics tail" "$METRICS_PATH" 20 "${incident_dir}/metrics_tail.txt"
  append_file_tail "monitor log tail" "$MONITOR_LOG" 200 "${incident_dir}/monitor_log_tail.txt"

  printf '%s\n' "$incident_dir"
}

run_codex_callback() {
  local incident_dir="$1"
  local reason="$2"
  local callback_log="${incident_dir}/codex_callback.log"
  local prompt_file="${incident_dir}/codex_prompt.txt"

  if [[ "$CODEX_AUTOFIX_ENABLED" != "1" ]]; then
    log "autofix disabled; incident bundle written to ${incident_dir}"
    return 0
  fi

  if [[ ! -x "$CODEX_BIN" ]]; then
    log "codex callback skipped because CODEX_BIN is not executable: ${CODEX_BIN}"
    return 1
  fi

  cat > "$prompt_file" <<EOF
You are a Codex instance invoked by an unattended monitor for a PhylaFlow training run.

The monitored run appears to have crashed or exited unexpectedly.

Run name:
${RUN_NAME}

Reason observed by monitor:
${reason}

Workspace:
${WORKDIR}

Incident bundle:
${incident_dir}

Exact training config:
${CONFIG_PATH}

Primary training log:
${RUN_LOG}

Expected sample metrics JSONL:
${METRICS_PATH}

Checkpoint directory:
${CHECKPOINT_DIR}

Your task:
1. Inspect the incident bundle, the full training log, the config, and the relevant repo code.
2. Identify why the run crashed or exited unexpectedly.
3. Patch the code or config if a clear fix is needed. Preserve unrelated user changes.
4. Relaunch the run after the fix. Use the same config unless your investigation shows the config itself needs a targeted correction.
5. Prefer the CUDA_VISIBLE_DEVICES value saved in ${incident_dir}/last_run_env.env if it exists.
6. Write a concise incident report to ${incident_dir}/codex_incident_report.md with cause, files changed, verification, and the relaunch command/PID.

Do not wait for long training progress after relaunch. Confirm the process starts and the log begins moving, then exit.
EOF

  log "starting codex callback for incident ${incident_dir}"
  (
    cd "$WORKDIR" || exit 1
    if command -v timeout >/dev/null 2>&1; then
      timeout "$CODEX_TIMEOUT_SECONDS" "$CODEX_BIN" exec \
        --dangerously-bypass-approvals-and-sandbox \
        --ask-for-approval never \
        --sandbox danger-full-access \
        -C "$WORKDIR" \
        < "$prompt_file"
    else
      "$CODEX_BIN" exec \
        --dangerously-bypass-approvals-and-sandbox \
        --ask-for-approval never \
        --sandbox danger-full-access \
        -C "$WORKDIR" \
        < "$prompt_file"
    fi
  ) >> "$callback_log" 2>&1
  local status=$?
  log "codex callback finished with status ${status}; log ${callback_log}"
  return "$status"
}

detect_log_error() {
  [[ -f "$RUN_LOG" ]] || return 1
  tail -n 250 "$RUN_LOG" |
    grep -Eiq 'Traceback \(most recent call last\)|RuntimeError:|ValueError:|KeyError:|IndexError:|CUDA out of memory|killed|segmentation fault|Exception:'
}

log_metrics_progress() {
  [[ -f "$METRICS_PATH" ]] || return 0

  local size
  local lines
  size="$(stat -c '%s' "$METRICS_PATH" 2>/dev/null || printf 0)"
  lines="$(wc -l < "$METRICS_PATH" 2>/dev/null || printf 0)"

  if [[ "$size" != "${LAST_METRICS_SIZE:-0}" || "$lines" != "${LAST_METRICS_LINE_COUNT:-0}" ]]; then
    LAST_METRICS_SIZE="$size"
    LAST_METRICS_LINE_COUNT="$lines"
    save_state
    log "metrics updated: ${METRICS_PATH} has ${lines} line(s), ${size} byte(s)"
  fi
}

maybe_handle_incident() {
  local reason="$1"
  local pid="${2:-}"
  local key="${reason}:${LAST_SEEN_PID:-none}:$(stat -c '%Y' "$RUN_LOG" 2>/dev/null || printf 0):$(stat -c '%s' "$RUN_LOG" 2>/dev/null || printf 0)"
  local now
  now="$(date +%s)"

  if [[ "$key" == "${LAST_INCIDENT_KEY:-}" && $((now - ${LAST_CODEX_AT:-0})) -lt "$CRASH_COOLDOWN_SECONDS" ]]; then
    return 0
  fi

  LAST_INCIDENT_KEY="$key"
  LAST_CODEX_AT="$now"
  save_state

  local incident_dir
  incident_dir="$(build_incident_bundle "$reason" "$pid")"
  log "incident captured: ${reason}; bundle ${incident_dir}"
  run_codex_callback "$incident_dir" "$reason" || true
}

log "monitor starting for ${RUN_NAME}"
log "poll=${POLL_SECONDS}s codex_autofix=${CODEX_AUTOFIX_ENABLED} config=${CONFIG_PATH}"

load_state

while true; do
  pid="$(find_run_pid)"

  if [[ -n "$pid" ]]; then
    if [[ "$pid" != "${LAST_SEEN_PID:-}" ]]; then
      LAST_SEEN_PID="$pid"
      save_state
      log "observed running training process pid=${pid}"
    fi
    record_live_env "$pid"
    log_metrics_progress
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ -n "${LAST_SEEN_PID:-}" ]]; then
    if detect_log_error; then
      maybe_handle_incident "process_absent_after_error_log" ""
    else
      maybe_handle_incident "process_absent_without_recent_error" ""
    fi
  elif [[ -f "$RUN_LOG" ]]; then
    log_mtime="$(stat -c '%Y' "$RUN_LOG" 2>/dev/null || printf 0)"
    now="$(date +%s)"
    if [[ "$log_mtime" != "0" && $((now - log_mtime)) -gt "$STALE_LOG_SECONDS" ]]; then
      maybe_handle_incident "no_pid_and_log_stale" ""
    fi
  else
    maybe_handle_incident "no_pid_and_missing_log" ""
  fi

  sleep "$POLL_SECONDS"
done
