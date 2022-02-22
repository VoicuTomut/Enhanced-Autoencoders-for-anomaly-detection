"""
This module contains all the decoders.
The most basic decoder is the inverse of an encoder,
but we can also have parametrized decoders or even decoders with other dimensions.
"""
from .base import decoder_adjoint
from .classic_parametrized_decoder import d1_classic
