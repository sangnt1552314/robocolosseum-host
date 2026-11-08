#!/bin/bash

export LAUNCHER_API_TOKEN=""

python pbs_remote_inference/local/run_robot.py \
  --model molmoact2 \
  --embodiment so101 \
  --follower-port /dev/tty.usbmodem5C821064861 \
  --wrist-camera 0 \
  --external-camera 1 \
  --instruction "stack the white bowls" \
  --launcher-url "<LAUNCHER_URL>" \
  --callback-url "<CALLBACK_URL>" \
  --launcher-token "$LAUNCHER_API_TOKEN"