"""Ember — an O(V+D) embedding optimizer. Adam quality on the token-embedding table at
~1500x less optimizer state. `Ember` is distributed-ready; `EmberReference` is the
single-device spec."""
from .ember import Ember
from .ember_reference import EmberReference
from .param_utils import split_embedding_params

__all__ = ["Ember", "EmberReference", "split_embedding_params"]
__version__ = "0.1.0"
