#!/bin/bash
# Serve BOTH robots from ONE interactive GPU job, for when batch jobs won't start (the interactive
# queue runs one job per user, up to 2 GPUs and 2 h). so101's models go on the job's first GPU and
# yam's on the second: the ten need ~143 GB (DETAILS.md), more than one H200. Each robot gets its
# usual tailnet node and ports, so relays and run_nscc_eval.sh work unchanged:
# colosseum-so101:9101-9105 and colosseum-yam:9111-9115.
#
#   qsub -I -P CFP05-SF-002 -l select=1:ngpus=2 -l walltime=02:00:00   # 2 GPUs
#   cd /scratch/e1583535/projects/robocolosseum-hosted
#   bash pbs_remote_inference/pbs/interactive_one_for_all.sh
#
# It runs interactive.sh once per robot (READY lines, tailnet join and logout). Ctrl-C stops both.
# SO101_MODELS / YAM_MODELS override the model lists; keep the order, it fixes the ports.
# Logs: logs/pbs_remote_inference/<robot>/<family>/*.<jobid>-<robot>.log
set -e
cd "$(dirname "$0")/../.."    # repo root
SO101_MODELS=${SO101_MODELS:-molmoact2:pi05:groot:g05:lingbot}
YAM_MODELS=${YAM_MODELS:-molmoact2:pi05:groot:lingbot:g05}

mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader 2> /dev/null)
if [ ${#GPUS[@]} -lt 2 ]; then
    echo "ERROR: this job has ${#GPUS[@]} GPU(s); so101 + yam need 2. Start it with -l select=1:ngpus=2," >&2
    echo "       or serve one robot with interactive.sh." >&2
    exit 1
fi

# Separate job IDs keep the two robots' tailscaled sockets and log files apart.
JOB=${PBS_JOBID:-interactive-$$}
PIDS=()
SERVE_GPU=${GPUS[0]} PBS_JOBID=$JOB-so101 bash pbs_remote_inference/pbs/interactive.sh so101 "$SO101_MODELS" tailscale &
PIDS+=($!)
SERVE_GPU=${GPUS[1]} PBS_JOBID=$JOB-yam bash pbs_remote_inference/pbs/interactive.sh yam "$YAM_MODELS" tailscale &
PIDS+=($!)
echo "so101 on GPU ${GPUS[0]}, yam on GPU ${GPUS[1]}; READY lines follow (1-5 min)."

# Background children of a script ignore Ctrl-C, so forward it as TERM: interactive.sh then stops
# its servers and logs its tailnet node out.
trap 'echo "Stopping so101 and yam..."; kill -TERM "${PIDS[@]}" 2> /dev/null; wait; exit 0' INT TERM
wait
