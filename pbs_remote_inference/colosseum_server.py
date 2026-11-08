#!/usr/bin/env python3
"""Serve one VLA policy over HTTP for Colosseum Policy Server.

Loads the model once through the project's existing adapters, runs one warm-up
inference, then answers on a single-threaded HTTP server (one request at a time):

    GET  /healthz -> 200 {"status": "ok"}
    POST /act     -> 200 {"actions": [[...], ...]}  or  500 {"error": "..."}

``--robot`` selects the request layout (Colosseum HTTP contract):

    franka  external_cam, wrist_cam            state/action: 7 joints (rad), gripper
    so101   external_cam, wrist_cam            state/action: 5 joints (deg), gripper [0, 100]
    yam     top_cam, left_cam, right_cam       state/action: left 6 + gripper, right 6 + gripper

Images are uint8 RGB (H, W, 3) at native resolution; the model's processor
resizes them. Each action row is an absolute joint target in the state layout.
The adapter sees every image under its request field name, plus
``joints`` / ``gripper`` (yam: the whole 14-value state as ``joints``).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pbs_remote_inference.policies.registry import create_local_policy, has_local_policy  # noqa: E402
from robocolosseum.config import load_app_config  # noqa: E402
from robocolosseum.policies.registry import create_policy  # noqa: E402

# robot -> (request image fields, state/action size)
ROBOTS = {
    "franka": (("external_cam", "wrist_cam"), 8),
    "so101": (("external_cam", "wrist_cam"), 6),
    "yam": (("top_cam", "left_cam", "right_cam"), 14),
}


def decode(value):
    return np.frombuffer(
        base64.b64decode(value["__numpy__"]), dtype=np.dtype(value["dtype"])
    ).reshape(value["shape"])


def load(config_path, checkpoint, device):
    config = load_app_config(config_path)
    if checkpoint:
        config.policy.checkpoint = checkpoint
    if device:
        config.policy.device = device
    name = config.policy.name
    policy = create_local_policy(name, config) if has_local_policy(name) else create_policy(name, config)
    policy.load()
    return policy, config


def act(policy, config, robot, images, state, instruction, num_steps=None):
    if num_steps is not None and hasattr(policy, "num_steps"):
        policy.num_steps = int(num_steps)  # denoising steps, for adapters that expose it
    dim = ROBOTS[robot][1]
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.size != dim:
        raise ValueError(f"state has {state.size} values, expected {dim}")
    single_arm = robot != "yam"
    observation = SimpleNamespace(
        instruction=instruction,
        state=SimpleNamespace(
            **images,
            joints=state[:-1] if single_arm else state,
            gripper=state[-1:] if single_arm else None,
        ),
    )
    actions = np.asarray(policy.predict(observation), dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    actions = actions[: config.runtime.max_horizon]
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != dim:
        raise ValueError(f"policy returned shape {actions.shape}, expected (N, {dim})")
    if not np.isfinite(actions).all():
        raise ValueError("policy returned NaN/Inf actions")
    return actions


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self.reply(200, {"status": "ok"})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/act":
            self.reply(404, {"error": "not found"})
            return
        try:
            start = time.perf_counter()
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            robot = self.server.robot
            actions = act(
                self.server.policy,
                self.server.config,
                robot,
                {name: decode(req[name]) for name in ROBOTS[robot][0]},
                decode(req["state"]),
                req["instruction"],
                req.get("num_steps"),
            )
            print(f"/act {actions.shape} in {time.perf_counter() - start:.3f}s", file=sys.stderr)
            self.reply(200, {"actions": actions.tolist()})
        except Exception as exc:
            traceback.print_exc()
            self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    parser.add_argument("--config", required=True, help="Policy YAML, e.g. configs/so101/molmoact2.yaml")
    parser.add_argument("--checkpoint", help="Override policy.checkpoint from the YAML")
    parser.add_argument("--device", help="Override policy.device from the YAML (e.g. cuda:0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    args = parser.parse_args()

    policy, config = load(args.config, args.checkpoint, args.device)
    fields, dim = ROBOTS[args.robot]
    images = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in fields}
    start = time.perf_counter()
    actions = act(policy, config, args.robot, images, np.zeros(dim, np.float32), "warm up")
    print(f"Warm-up {actions.shape} in {time.perf_counter() - start:.3f}s", flush=True)

    server = HTTPServer((args.host, args.port), Handler)
    server.policy, server.config, server.robot = policy, config, args.robot
    print(f"Serving {config.policy.name} ({args.robot}) on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
