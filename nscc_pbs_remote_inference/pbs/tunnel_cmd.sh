#!/bin/bash
# Print the SSH command that makes a running colosseum job's models reachable on the Policy Server
# machine as http://127.0.0.1:<local port>. Colosseum Policy Server only accepts loopback endpoints.
# Run on an NSCC login node:
#
#   bash nscc_pbs_remote_inference/pbs/tunnel_cmd.sh <jobid> [first local port, default 9201]
#
# Then run the printed command on the Policy Server machine and keep it open. It jumps through the
# login node to the GPU node and maps each model port to a local port. Replace <nscc-login> with
# your login host. SSH forwarding to ASPIRE 2B GPU nodes has not been tested.
set -euo pipefail
JOB=${1:?usage: tunnel_cmd.sh <jobid> [first local port]}
LOCAL=${2:-9201}
qstat -f -F json "$JOB" | python3 -c "
import json, socket, sys
job = list(json.load(sys.stdin)['Jobs'].values())[0]
if job['job_state'] != 'R':
    sys.exit(f'job is {job[\"job_state\"]}, not running yet')
node = job['exec_host'].split('/')[0]
ip = socket.gethostbyname(node)
env = job['Variable_List']
robot = env.get('EMBODIMENT', '?')
name = job['Job_Name']
base = int(env.get('BASE_PORT') or {'so101': 9101, 'yam': 9111, 'franka': 9121}.get(robot, 9101))
if env.get('MODELS'):                      # cohost.pbs: consecutive ports in MODELS order
    models = [(m, base + i) for i, m in enumerate(env['MODELS'].split(':'))]
else:                                      # <family>.pbs: family from the job name colosseum-<family>-nscc
    models = [(name.removeprefix('colosseum-').removesuffix('-nscc'), int(env.get('PORT') or base))]
local = $LOCAL
forwards = ' '.join(f'-L {local + i}:127.0.0.1:{port}' for i, (_, port) in enumerate(models))
print(f'# job $JOB on {node} ({ip}), robot {robot}. Run on the Policy Server machine:')
print(f'ssh -N -o GSSAPIAuthentication=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \\\\')
print(f'    -J $USER@<nscc-login> {forwards} $USER@{ip}')
print('# Policy Server endpoints:')
for i, (model, port) in enumerate(models):
    print(f'#   {robot}/{model}: http://127.0.0.1:{local + i}')
"
