"""
Model architecture for trajectory prediction

Base Classes:
- BaseTrajectoryModel: Abstract base for all models
- ContinuousTimeModel: Base for continuous-time models (SB, OT)
- TimeConditionedModel: Base for discrete time-conditioned models (VAE, BatchOT)

Concrete Models:
- SchrodingerBridgeModel: Base SB model
- MLPlus_SchrodingerBridgeModel: Enhanced SB model with residual connections
- OptimalTransportModel: Optimal transport based model
- RegularizedOTModel: OT model with gradient penalty
- VAEModel: Variational autoencoder model
- ConditionalVAEModel: Time-conditional VAE for cell state transition
- BatchOTModel: Batch optimal transport model
"""

from .base_model import BaseTrajectoryModel, ContinuousTimeModel, TimeConditionedModel
from .sb_model import SchrodingerBridgeModel
from .sb_model_mlplus import MLPlus_SchrodingerBridgeModel
from .ot_model import OptimalTransportModel, RegularizedOTModel
from .batch_ot_model import BatchOTModel
from .vae_model import VAEModel
from .c_vae_model import ConditionalVAEModel

__all__ = [
    'BaseTrajectoryModel',
    'ContinuousTimeModel',
    'TimeConditionedModel',
    'SchrodingerBridgeModel',
    'MLPlus_SchrodingerBridgeModel',
    'OptimalTransportModel',
    'RegularizedOTModel',
    'BatchOTModel',
    "VAEModel",
    "ConditionalVAEModel",
]
