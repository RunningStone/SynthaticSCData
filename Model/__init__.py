"""
Model architectures for trajectory prediction
"""

from .ot_model import OptimalTransportModel
from .sb_model import SchrodingerBridgeModel
from .vae_model import VAEModel

__all__ = [
    "OptimalTransportModel",
    "SchrodingerBridgeModel",
    "VAEModel",
]
