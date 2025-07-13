"""
Pinball Environment Library
"""

from .pinball import PinballEnv
from .pinball_configs import available_configs
from .feature_construction.tile_coder import PinballTileCoder

__version__ = "0.1.0"
__all__ = ["PinballEnv", "available_configs", "PinballTileCoder"]
