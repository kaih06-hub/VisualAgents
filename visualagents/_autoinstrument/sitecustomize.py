"""Loaded automatically by Python because `visualagents run` prepends this folder to PYTHONPATH.

Installs VisualAgents' framework hooks, then runs any sitecustomize this one shadows.
"""

import os
import sys


def _visualagents_boot():
    if os.environ.get("VISUALAGENTS_DISABLE") == "1":
        return
    try:
        import visualagents_runtime

        visualagents_runtime.instrument()
    except Exception:
        pass


def _visualagents_chain():
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in list(sys.path):
        try:
            if not entry or os.path.abspath(entry) == here:
                continue
            candidate = os.path.join(entry, "sitecustomize.py")
            if os.path.isfile(candidate):
                import importlib.util

                spec = importlib.util.spec_from_file_location("_visualagents_shadowed_sitecustomize", candidate)
                if spec and spec.loader:
                    spec.loader.exec_module(importlib.util.module_from_spec(spec))
                return
        except Exception:
            return


_visualagents_boot()
_visualagents_chain()
