"""Run at interpreter startup by the `visualagents_hook.pth` file that `visualagents hook` installs.

Stdlib only, Python 3.8+, and it must never break the host program: every failure is swallowed.
The .pth line executes this file with VISUALAGENTS_PROJECT set to the hooked folder.
"""

import json
import os
import sys


def _start(project):
    if os.environ.get("VISUALAGENTS_DISABLE") == "1" or os.environ.get("VISUALAGENTS_URL"):
        return  # turned off, or `visualagents run` is already instrumenting this process
    home = os.environ.get("VISUALAGENTS_HOME") or os.path.join(os.path.expanduser("~"), ".visualagents")
    daemon_file = os.path.join(home, "daemon.json")
    info = {}
    try:
        with open(daemon_file, encoding="utf-8") as fh:
            info = json.load(fh)
    except Exception:
        pass
    rt = sys.modules.get("visualagents_runtime")
    if rt is None:
        import importlib.util

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "_autoinstrument", "visualagents_runtime.py")
        spec = importlib.util.spec_from_file_location("visualagents_runtime", path)
        rt = importlib.util.module_from_spec(spec)
        sys.modules["visualagents_runtime"] = rt
        spec.loader.exec_module(rt)
    rt.configure(url=info.get("url"), token=info.get("token"), project=project, daemon_file=daemon_file, lazy=True)
    rt.instrument()
    if os.environ.get("VISUALAGENTS_CAPTURE_OUTPUT", "1") != "0":
        rt.capture_output()


try:
    _start(globals().get("VISUALAGENTS_PROJECT") or os.getcwd())
except Exception:
    pass
