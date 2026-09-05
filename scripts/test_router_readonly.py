#!/usr/bin/env python3
"""Read-only Colosseum router test.

Connects to the router with the official SDK, inspects real observations, and
prints their structure. It NEVER calls ``send_action`` — nothing can move the
robot. Use this to verify the real DROID observation format (camera field
names, shapes, dtypes, joint/gripper layout, action spaces).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robocolosseum.config import load_router_config  # noqa: E402


def _describe(obs) -> None:
    print("=" * 60)
    print(f"session_id: {obs.session_id}")
    print(
        f"sequence: {obs.sequence}  control_step: {obs.control_step}  "
        f"deadline_ms: {obs.deadline_ms}"
    )
    print(f"instruction: {obs.instruction!r}")

    joints = obs.state.joints
    gripper = obs.state.gripper
    print(
        f"joints: shape={None if joints is None else joints.shape} "
        f"dtype={None if joints is None else joints.dtype}"
    )
    print(f"gripper: value={gripper}")

    print("raw image frames:")
    for img in obs.images:
        print(
            f"  id={img.image_id} encoding={img.encoding} "
            f"{img.width}x{img.height} bytes={len(img.data)}"
        )
    print("decoded standard cameras:")
    for name in ("left_image", "right_image", "head_image"):
        arr = getattr(obs.state, name)
        detail = None if arr is None else (arr.shape, str(arr.dtype))
        print(f"  state.{name}: {detail}")


def _save_images(obs, index: int, out_dir: Path) -> None:
    """Save each available decoded camera (RGB NumPy array) as a PNG."""
    from PIL import Image

    for name, suffix in (
        ("left_image", "left"),
        ("right_image", "right"),
        ("head_image", "head"),
    ):
        arr = getattr(obs.state, name)
        if arr is None:
            print(f"  {name} missing; skipping")
            continue
        path = out_dir / f"obs_{index:03d}_{suffix}.png"
        # SDK arrays are already RGB, which is the order Pillow expects.
        Image.fromarray(arr).save(path)
        print(f"Saved {name} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-config", default=None)
    parser.add_argument(
        "--num", type=int, default=1, help="observations to inspect before exiting"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--save-images",
        default=None,
        metavar="DIR",
        help="save decoded camera images from each observation into DIR",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO", format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger("router-readonly")

    save_dir = None
    if args.save_images is not None:
        save_dir = Path(args.save_images)
        save_dir.mkdir(parents=True, exist_ok=True)
        log.info("Saving camera images to %s", save_dir)

    from colosseum_policy_server import ColosseumPolicySDK, SDKConnectionError

    router = load_router_config(args.router_config)
    log.info("Connecting to router (read-only)...")
    with ColosseumPolicySDK(router_url=router.url, token=router.token) as sdk:
        log.info("Connected as server_id=%s", sdk.server_id)

        seen = 0
        while seen < args.num:
            try:
                obs = sdk.get_obs(timeout=args.timeout)
            except TimeoutError:
                log.info("No observation within %.0fs; exiting", args.timeout)
                break
            except SDKConnectionError as exc:
                log.info("Connection closed: %s", exc)
                break

            _describe(obs)
            if save_dir is not None:
                _save_images(obs, seen, save_dir)
            if sdk.robot is not None:
                log.info(
                    "robot_type=%s joint_count=%s has_gripper=%s control_hz=%s",
                    sdk.robot.robot_type,
                    sdk.robot.joint_count,
                    sdk.robot.has_gripper,
                    sdk.robot.control_hz,
                )
                log.info(
                    "selected_action_space=%s action_spaces=%s",
                    sdk.selected_action_space,
                    dict(sdk.robot.action_spaces),
                )
            seen += 1

        log.info("Read-only test complete. No actions were sent.")


if __name__ == "__main__":
    main()
