"""Serve a VLA policy on a PBS GPU node over HTTP for Colosseum Policy Server.

``colosseum_server.py`` loads one policy through the existing adapters
(``policies/`` first, then ``src/robocolosseum/policies``) and answers
``GET /healthz`` and ``POST /act``. See README.md.
"""
