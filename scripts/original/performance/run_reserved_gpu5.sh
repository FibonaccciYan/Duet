#!/usr/bin/env bash
# n33 GPU 5 only. Always restore the user's existing reservation configuration.
set -u
HP=/data0/gs/entangle_parallel_capture/.venv/bin/python
TOOL=/home/gs/.cache/codex-tools/gpu_reserver.py
STATE=/home/gs/.cache/codex-gpu-reserver-n33
restore_needed=1
restore() {
    if [ "$restore_needed" -eq 0 ]; then return 0; fi
    # GPU utilization sampling can remain high just after process exit.
    for attempt in 1 2 3; do
        if "$HP" "$TOOL" --state-dir "$STATE" restart; then
            if "$HP" "$TOOL" --state-dir "$STATE" status; then
                restore_needed=0
                return 0
            fi
        fi
        sleep 2
    done
    echo HOLDER_RESTORE_FAILED >&2
    return 92
}
trap restore EXIT
trap 'exit 130' HUP INT TERM
"$HP" "$TOOL" --state-dir "$STATE" stop || exit 91
if "$HP" "$TOOL" --state-dir "$STATE" status >/dev/null 2>&1; then
    echo HOLDER_STILL_RUNNING >&2
    exit 91
fi
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
"$@"
test_rc=$?
restore || exit 92
trap - EXIT HUP INT TERM
exit "$test_rc"
