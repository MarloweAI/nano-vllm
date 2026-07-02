"""Compatibility shim for the old GPT-OSS-specific nano timing module.

New code should use ``nanovllm.mock.timing.analytical`` and pass model/hardware
keys from ``analytical_backend``. This module remains importable for validation
tools that still reference the historical name.
"""

from nanovllm.mock.timing.analytical import *  # noqa: F401,F403
