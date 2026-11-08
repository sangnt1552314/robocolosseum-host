#!/bin/bash
# Print the SSH command that makes a running colosseum job's models reachable on the
# Policy Server machine as http://127.0.0.1:<local port>. Colosseum Policy Server only
# accepts loopback endpoints. Run on Hopper:
#
#   bash pbs_remote_inference/pbs/tunnel_cmd.sh <jobid> [first local port, default 9201]
#
# Then run the printed command on the Policy Server machine and keep it open. It jumps
# through the login node to the GPU node (port forwarding only; Hopper GPU nodes kill
# remote commands but allow forwards) and maps each model port to a local port.
set -e
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
if env.get('MODELS'):                      # cohost.pbs: ports 9101, 9102, ... in MODELS order
    models = [(m, 9101 + i) for i, m in enumerate(env['MODELS'].split(':'))]
else:                                      # <family>.pbs: family from the job name colosseum-<family>
    models = [(name.removeprefix('colosseum-'), int(env.get('PORT') or 9101))]
local = $LOCAL
forwards = ' '.join(f'-L {local + i}:127.0.0.1:{port}' for i, (_, port) in enumerate(models))
print(f'# job $JOB on {node} ({ip}), robot {robot}. Run on the Policy Server machine:')
print(f'ssh -N -o GSSAPIAuthentication=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \\\\')
print(f'    -J $USER@<hopper-login> {forwards} $USER@{ip}')
print('# Policy Server endpoints:')
for i, (model, port) in enumerate(models):
    print(f'#   {robot}/{model}: http://127.0.0.1:{local + i}')
"
