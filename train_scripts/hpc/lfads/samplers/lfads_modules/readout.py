"""
lfads-torch modules needed for unconditional sampling

Author: Andrew Sedler
Sources: 
  - https://github.com/arsedler9/lfads-torch/blob/d8b4b3ba87a49fd74a3c06afb3ec1b695c6a2227/lfads_torch/modules/readin_readout.py
"""

import math

import numpy as np
from torch import nn


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)
