"""
model.py  -  Spatial CNN Q-network for Chain Reaction.

Architecture
------------
  Input  : (batch, 4, H, W)  — four-channel board, any H and W
  Block 1: Conv2d(4→32,  3×3, padding=1) + BatchNorm + ReLU
  Block 2: Conv2d(32→64, 3×3, padding=1) + BatchNorm + ReLU
  Block 3: Conv2d(64→64, 3×3, padding=1) + BatchNorm + ReLU
  Head   : Conv2d(64→1,  1×1)            — one Q-value per cell
  Output : flatten → (batch, H*W)

The four input channels (always from current player's perspective):
  0  my orb counts    / 4
  1  enemy orb counts / 4
  2  critical mass    / 4   (static geometry)
  3  primed map       — +1.0 my primed, -1.0 enemy primed, 0.0 else

The primed map gives the CNN direct spatial information about danger:
  +1 next to -1  =>  exposure (danger if left, opportunity if blasted)
  +1 next to +1  =>  your chain
  -1 next to -1  =>  enemy chain nearby

Grid-size independence
----------------------
Same trained weights work on any board size — pass a differently-sized
input tensor and the conv layers slide across it unchanged.
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Fully-convolutional Q-network — works on any board size.

    Args:
        rows, cols : board dimensions (used only for reshape inside forward)
        channels   : number of feature channels in the hidden conv layers
    """

    def __init__(self, rows: int, cols: int, channels: int = 64):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.conv = nn.Sequential(
            # Block 1: detect basic features across all 4 input channels
            nn.Conv2d(4,        32,       kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2: combine neighbourhood info (chains, exposure patterns)
            nn.Conv2d(32,       channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            # Block 3: higher-order patterns (trigger timing, board control)
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            # Head: collapse features → 1 Q-value per cell
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-cell Q-values.

        Args:
            x : (batch, 4*rows*cols) flat tensor   <- from replay buffer
                OR (batch, 4, rows, cols) image tensor

        Returns:
            q : (batch, rows*cols) Q-values, one per board cell
        """
        b = x.size(0)
        if x.dim() == 2:
            x = x.view(b, 4, self.rows, self.cols)
        return self.conv(x).view(b, -1)
