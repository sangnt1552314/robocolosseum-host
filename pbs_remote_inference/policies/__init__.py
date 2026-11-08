"""Local policy adapters for colosseum_server.py.

Adapters here extend the shared ones in ``src/robocolosseum/policies`` (which
target DROID) for other embodiments, without modifying them. The server looks a
policy name up in :data:`registry.LOCAL_POLICY_REGISTRY` first and falls back to
the shared registry. Add a model by dropping a module here and registering one line.
"""
