#!/bin/bash

export LAUNCHER_API_TOKEN="4e03ba5488f3eafebf5217b5ef8f89f6ab02946c932cc106051ce1c6ee75e204"

python pbs_remote_inference/local/run_robot.py \
  --model molmoact2 \
  --embodiment so101 \
  --follower-port /dev/tty.usbmodem5C821064861 \
  --wrist-camera 0 \
  --external-camera 1 \
  --instruction "stack the white bowls" \
  --launcher-url https://bucktooth-deck-suitably.ngrok-free.dev \
  --callback-url wss://uncheating-rolanda-noncaustic.ngrok-free.dev/worker \
  --launcher-token "$LAUNCHER_API_TOKEN"