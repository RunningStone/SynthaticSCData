"""
Model architecture for trajectory prediction
- SchrodingerBridgeModel: Base model
- MLPlus_SchrodingerBridgeModel: Enhanced model with residual connections
"""

from .sb_model import SchrodingerBridgeModel
from .sb_model_mlplus import MLPlus_SchrodingerBridgeModel

__all__ = [
    "SchrodingerBridgeModel",
    "MLPlus_SchrodingerBridgeModel",
]
