"""Backward-compatible WHAM API entrypoint.

Prefer importing from `wham`:

    from wham import WHAMRunner, run_wham
"""

from wham.api import WHAMRunner, run_wham


class WHAM_API(WHAMRunner):
    """Compatibility alias for previous API class name."""


__all__ = ["WHAMRunner", "WHAM_API", "run_wham"]
