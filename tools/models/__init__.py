"""Network definitions used by the benchmark adapters.

Re-exported here: SDeepONet3D (CMP field DeepONet, cmp3d_sdeeponet), ConvAutoencoder
(conv_autoencoder) and the S4D kernel and layer (ssm). The KVS field DeepONet
(kvs_field_deeponet.KvsFieldDeepONet) is imported directly by its adapter.
"""

from .cmp3d_sdeeponet import SDeepONet3D
from .conv_autoencoder import ConvAutoencoder
from .ssm import S4DKernel, S4DLayer

__all__ = [
    "ConvAutoencoder",
    "S4DKernel",
    "S4DLayer",
    "SDeepONet3D",
]
