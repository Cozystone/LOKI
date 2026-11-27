"""Bounded local sensory memory."""
from .organ import LokiMemory

from .cortex import CorticalMemory, LongTermCortex
from .reverse import ReverseMemory
from .portable import PortableMemory
from .contextual import ContextMemory, Transition

from .body import BodyMemory

from .episodic import EpisodeMemory

from .episode_graph import GraphMemory

from .belief import BeliefModel

__all__ = ["BeliefModel", "LokiMemory", "LongTermCortex", "CorticalMemory", "ReverseMemory", "PortableMemory", "ContextMemory", "Transition", "BodyMemory", "EpisodeMemory", "GraphMemory"]

# Keep the v4-only import path free of the v12 Numba dependency.
def __getattr__(name):
    if name == "TraceStore":
        from .trace_store import TraceStore
        return TraceStore
    if name == "TraceAtlas":
        from .trace_atlas import TraceAtlas
        return TraceAtlas
    if name == "LokiV12":
        from .journey import LokiV12
        return LokiV12
    raise AttributeError(name)
