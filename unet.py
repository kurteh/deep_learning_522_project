"""
unet.py — Configurable U-Net for Spectral Analysis
====================================================
Designed so that:
  - Encoder/decoder are symmetric and easy to compare level-by-level
  - All conv weight matrices are easily extracted for SVD
  - Weights are saved in a structured dict keyed by (block, level, layer)

Usage
-----
    from unet import UNet, train_one_epoch, save_weights_for_analysis

    model = UNet(
        in_channels=1,
        out_channels=2,       # e.g. binary segmentation
        base_channels=64,     # width at the first encoder level
        depth=4,              # number of down/up-sampling levels
        bilinear=True,        # use bilinear upsampling vs. transposed conv
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DoubleConv(nn.Module):
    """
    Two consecutive (Conv → BatchNorm → ReLU) operations.
    This is the basic repeated unit in both encoder and decoder.

    Weight shapes (for SVD later):
        conv1.weight : (out_ch, in_ch, 3, 3)
        conv2.weight : (out_ch, out_ch, 3, 3)
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            ("bn1",   nn.BatchNorm2d(mid_channels)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            ("bn2",   nn.BatchNorm2d(out_channels)),
            ("relu2", nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder step: MaxPool → DoubleConv (halves spatial dims)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Decoder step: Upsample → concat skip → DoubleConv (doubles spatial dims).
    
    Skip connection arrives here as the second argument to forward().
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            # Bilinear upsample then halve channels with 1×1 conv
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # Learned transposed convolution (also interesting to SVD)
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if input is not perfectly divisible (odd spatial dims)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])

        # ── Skip connection concatenation ──
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    Configurable U-Net.

    Parameters
    ----------
    in_channels   : input image channels (1 = grayscale, 3 = RGB)
    out_channels  : number of segmentation classes
    base_channels : feature channels at the first encoder level (doubles each level)
    depth         : number of down/up-sampling stages (encoder levels excluding bottleneck)
    bilinear      : True  → bilinear upsampling (fewer params)
                    False → transposed conv (learnable upsampling, more to SVD)

    Architecture (depth=4, base=64)
    --------------------------------
    Input → [64] → [128] → [256] → [512] → [1024] (bottleneck)
                                         ↕  skip[3]
                              [512] ← Up(1024+512)
                         ↕  skip[2]
                   [256] ← Up(512+256)
              ↕  skip[1]
        [128] ← Up(256+128)
    ↕  skip[0]
    [64]  ← Up(128+64)
    → Output conv → logits
    """

    def __init__(
        self,
        in_channels:   int  = 1,
        out_channels:  int  = 2,
        base_channels: int  = 64,
        depth:         int  = 4,
        bilinear:      bool = True,
    ):
        super().__init__()
        self.depth    = depth
        self.bilinear = bilinear

        # ── Encoder ──────────────────────────────────────────────────────────
        # enc[0] : initial conv (no pooling)
        # enc[1..depth-1] : Down blocks
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(in_channels, base_channels))
        ch = base_channels
        for _ in range(depth - 1):
            self.encoder.append(Down(ch, ch * 2))
            ch *= 2

        # ── Bottleneck ───────────────────────────────────────────────────────
        factor = 2 if bilinear else 1
        self.bottleneck = Down(ch, ch * 2 // factor)
        ch = ch * 2 // factor

        # ── Decoder ──────────────────────────────────────────────────────────
        # Symmetric to encoder; skip from enc[depth-1] down to enc[0]
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            skip_ch = base_channels * (2 ** i)
            self.decoder.append(Up(ch + skip_ch, skip_ch // factor, bilinear))
            ch = skip_ch // factor

        self.decoder.append(Up(ch + base_channels, base_channels, bilinear))

        # ── Output ───────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode — store intermediate feature maps for skip connections
        skips = []
        for enc_block in self.encoder:
            x = enc_block(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode — consume skips in reverse order
        for dec_block, skip in zip(self.decoder, reversed(skips)):
            x = dec_block(x, skip)

        return self.out_conv(x)

    # ── Weight extraction helpers (for SVD) ──────────────────────────────────
 
    def get_weight_dict(self) -> dict:
        """
        Returns a flat dict of all conv weight tensors, keyed by human-readable names.
        Each value is a 2D numpy array (out_features × in_features) obtained by
        reshaping the 4D conv kernel (out, in, kH, kW) → (out, in*kH*kW).
 
        This 'unrolled' view is the natural matrix for computing singular values.
 
        Keys follow the pattern:
            "encoder_L{i}_conv{j}"   — encoder level i, conv layer j
            "bottleneck_conv{j}"
            "decoder_L{i}_conv{j}"   — decoder level i (from top of decoder)
        """
        weights = {}
 
        def extract_double_conv(prefix, double_conv: DoubleConv):
            for name in ["conv1", "conv2"]:
                w = getattr(double_conv.block, name).weight.detach().cpu()
                # Reshape (out, in, kH, kW) → (out, in*kH*kW)
                weights[f"{prefix}_{name}"] = w.reshape(w.shape[0], -1).numpy()
 
        # Encoder
        for i, block in enumerate(self.encoder):
            conv = block if isinstance(block, DoubleConv) else block.conv
            extract_double_conv(f"encoder_L{i}", conv)
 
        # Bottleneck
        extract_double_conv("bottleneck", self.bottleneck.conv)
 
        # Decoder
        for i, block in enumerate(self.decoder):
            extract_double_conv(f"decoder_L{i}", block.conv)
 
        return weights
 
    def save_weights_for_analysis(self, path: str):
        """
        Save structured weights to a .pt file.
        Loads back with:
            data = torch.load('weights.pt')
            data['weight_dict']   # flat dict of 2D numpy arrays
            data['model_state']   # full state dict to restore model
            data['config']        # hyperparameters
        """
        torch.save({
            "model_state": self.state_dict(),
            "weight_dict": self.get_weight_dict(),
            "config": {
                "depth":         self.depth,
                "bilinear":      self.bilinear,
            }
        }, path)
        print(f"Saved weights → {path}")