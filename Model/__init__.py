"""
Model architecture for trajectory prediction
- SchrodingerBridgeModel: Base model
- MLPlus_SchrodingerBridgeModel: Enhanced model with residual connections
- OptimalTransportModel: Optimal transport based model
- RegularizedOTModel: OT model with gradient penalty
- VAEModel: Variational autoencoder model
- ConditionalVAEModel: Conditional VAE for source->target mapping
"""

from .sb_model import SchrodingerBridgeModel
from .sb_model_mlplus import MLPlus_SchrodingerBridgeModel
from .ot_model import OptimalTransportModel, RegularizedOTModel
from .vae_model import VAEModel, ConditionalVAEModel

__all__ = [
    "SchrodingerBridgeModel",
    "MLPlus_SchrodingerBridgeModel",
    "OptimalTransportModel",
    "RegularizedOTModel",
    "VAEModel",
    "ConditionalVAEModel",
]
