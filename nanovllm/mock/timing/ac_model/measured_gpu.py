"""Compatibility path: this model lives in analytical_backend.ac_model.

Forwards the MODULE OBJECT so mutable globals (e.g. cs4_offload.CLOS_LAT_US)
stay shared across every import path.
"""
import sys

from analytical_backend.ac_model import measured_gpu as _shared

sys.modules[__name__] = _shared
