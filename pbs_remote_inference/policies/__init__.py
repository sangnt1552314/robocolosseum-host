"""Worker-local policy adapters (LeRobot checkpoints).

These adapters live inside ``pbs_remote_inference`` so the worker can load models
that need a loading path different from the ones in ``robocolosseum.policies``
(e.g. LeRobot fine-tuned checkpoints) without modifying the shared adapters.

The worker resolves a policy name against :data:`registry.LOCAL_POLICY_REGISTRY`
first and falls back to the shared ``robocolosseum`` registry when absent. Add a
new model by dropping an adapter module here and registering one line.
"""
