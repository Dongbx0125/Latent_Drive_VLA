from __future__ import annotations

from pathlib import Path
from typing import Any

_client: Any = None
_backend = "none"


def init(name=None, dir=None, project=None, entity=None, group=None, **kwargs):
    """SwanLab compatibility layer with legacy tracker-style init arguments."""
    global _client, _backend
    import swanlab

    run_kwargs = {}
    if project is not None:
        run_kwargs["project"] = project
    if name is not None:
        run_kwargs["name"] = name
    if entity is not None:
        run_kwargs["workspace"] = entity
    if dir is not None:
        logdir = Path(dir)
        if logdir.name != "swanlab":
            logdir = logdir / "swanlab"
        run_kwargs["logdir"] = str(logdir)
    if group is not None:
        run_kwargs["description"] = group
    run_kwargs.update(kwargs)

    try:
        swanlab.init(**run_kwargs)
    except TypeError:
        fallback_kwargs = {
            "project": run_kwargs.get("project"),
            "experiment_name": run_kwargs.get("name"),
            "logdir": run_kwargs.get("logdir"),
        }
        if run_kwargs.get("workspace") is not None:
            fallback_kwargs["workspace"] = run_kwargs["workspace"]
        if run_kwargs.get("description") is not None:
            fallback_kwargs["description"] = run_kwargs["description"]
        swanlab.init(**fallback_kwargs)

    _client = swanlab
    _backend = "swanlab"


def log(metrics, step=None):
    if _client is None:
        return
    if step is None:
        _client.log(metrics)
        return
    try:
        _client.log(metrics, step=step)
    except TypeError:
        _client.log(metrics)


def finish():
    if _client is None:
        return
    _client.finish()


def client():
    return _client
