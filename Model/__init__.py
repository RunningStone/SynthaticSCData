"""
Model architecture for trajectory prediction
- SchrodingerBridgeModel: Base model
- MLPlus_SchrodingerBridgeModel: Enhanced model with residual connections
- OptimalTransportModel: Optimal transport based model
- RegularizedOTModel: OT model with gradient penalty
- VAEModel: Variational autoencoder model
- ConditionalVAEModel: Time-conditional VAE for cell state transition
- BatchOTModel: Batch optimal transport model
"""

from .sb_model import SchrodingerBridgeModel
from .sb_model_mlplus import MLPlus_SchrodingerBridgeModel
from .ot_model import OptimalTransportModel, RegularizedOTModel
from .batch_ot_model import BatchOTModel
from .vae_model import VAEModel
from .c_vae_model import ConditionalVAEModel

__all__ = [
    'SchrodingerBridgeModel',
    'MLPlus_SchrodingerBridgeModel',
    'OptimalTransportModel',
    'RegularizedOTModel',
    'BatchOTModel',
    "VAEModel",
    "ConditionalVAEModel",
]
