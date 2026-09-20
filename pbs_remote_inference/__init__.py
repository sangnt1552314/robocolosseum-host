"""PBS remote inference workflow (launcher-orchestrated).

A robot user calls a lightweight launcher API to request a model. The launcher
selects the model's PBS script and submits a GPU job with ``qsub``. The GPU
worker loads the existing policy adapter, then connects OUTWARD (WSS) to the
user's own local WebSocket server and exchanges observations/action chunks
directly. The launcher is never in the observation/action data path.

This package is self-contained and does not depend on ``slurm_job_launcher_server``
or any external policy-server SDK; it only reuses the policy adapters under
``src/robocolosseum/policies``.
"""
