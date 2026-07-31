#!/usr/bin/env bash
set -euo pipefail

repository=/root/autodl-tmp/kitti_project
python=/root/autodl-tmp/venvs/kitti-yolo/bin/python3
job_dir=/root/autodl-tmp/jobs/multiseed-evidence
pid_file="$job_dir/pid.txt"
log_file="$job_dir/nohup.log"

cd "$repository"
test -x "$python"
git diff --quiet
git diff --cached --quiet

mkdir -p "$job_dir"
if [[ -f "$pid_file" ]]; then
    existing_pid=$(tr -d '[:space:]' <"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Evidence queue is already running as PID $existing_pid"
        exit 1
    fi
fi

commit=$(git rev-parse HEAD)
printf '%s\n' "$commit" >"$job_dir/commit.txt"
nohup "$python" -u "$repository/scripts/run_evidence_queue.py" \
    --repository-root "$repository" \
    --job-dir "$job_dir" \
    --device 0 \
    --python "$python" \
    >>"$log_file" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"

echo "Evidence queue started: commit=$commit pid=$pid"
echo "Live log: tail -f $log_file"
echo "Status:   cat $job_dir/status.json"
