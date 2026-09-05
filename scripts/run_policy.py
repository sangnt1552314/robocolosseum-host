#!/usr/bin/env python3
"""Entry point for the PBS policy worker.

Connects the common runner to the Colosseum router. Actions are NOT sent to the
robot unless ``--enable-action`` is passed (dry-run is the default for safety).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robocolosseum.config import load_app_config, load_router_config  # noqa: E402
from robocolosseum.runner import run_worker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--router-config",
        default=None,
        help="optional private YAML with url/token (env vars take precedence)",
    )
    parser.add_argument(
        "--enable-action",
        action="store_true",
        help="send actions to the robot (OFF by default: dry run)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_policy")

    config = load_app_config(args.config)
    if args.policy != config.policy.name:
        log.warning(
            "CLI --policy=%s overrides config policy.name=%s",
            args.policy,
            config.policy.name,
        )
        config.policy.name = args.policy

    router = load_router_config(args.router_config)

    if args.enable_action:
        log.warning("ACTION SENDING ENABLED - the robot may move.")
    else:
        log.info("Dry run: actions will be printed, not sent (default).")

    try:
        run_worker(config, router, enable_action=args.enable_action)
    except Exception as exc:  # noqa: BLE001 - top-level guard for a clean exit code
        log.exception("Policy worker failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
