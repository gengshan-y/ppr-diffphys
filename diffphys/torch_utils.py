import numpy as np
import pdb
import torch
from torch import nn

import sys, os

sys.path.append("%s/../../../../" % os.path.dirname(__file__))
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.nnutils.base import ScaleLayer
from lab4d.nnutils.time import TimeMLP


class TimeMLPOld(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        tscale=1.0,
        N_freqs=6,
        out_channels=3,
        skips=[4],
        activation=nn.ReLU(True),
    ):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        skips: add skip connection in the Dth layer
        """
        super(TimeMLPOld, self).__init__()
        self.D = D
        self.tscale = tscale
        self.embed = PosEmbedding(1, N_freqs)
        in_channels_xyz = self.embed.out_channels
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)

        # output layers
        self.pred = nn.Linear(W, out_channels)

        torch.manual_seed(8)  # to reproduce results
        torch.cuda.manual_seed(1)

    def get_vals(self, inx):
        if inx.dim() == 1:
            inx = inx.reshape(-1, 1)
        return self.forward(inx)

    def forward(self, inx):
        inx = inx * self.tscale
        inx = self.embed(inx)

        inx_ = inx
        for i in range(self.D):
            if i in self.skips:
                inx_ = torch.cat([inx, inx_], -1)
            inx_ = getattr(self, f"xyz_encoding_{i+1}")(inx_)

        out = self.pred(inx_)
        return out


class TimeMLPWarpper(TimeMLP):
    """Encode arbitrary scalar over time with an MLP

    Args:
        init_vals: (N,...) initial value of the scalar
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        out_channels (int): Number of output channels
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        num_frames,
        D=5,
        W=256,
        num_freq_t=6,
        out_channels=1,
        skips=[1, 2, 3, 4],
        activation=nn.ReLU(True),
        output_scale=1.0,
    ):
        # create info map for time embedding
        frame_info = {
            "frame_offset": np.asarray([0, num_frames]),
            "frame_mapping": list(range(num_frames)),
            "frame_offset_raw": np.asarray([0, num_frames]),
        }
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        # output layers
        self.head = nn.Sequential(
            nn.Linear(W, out_channels),
            ScaleLayer(output_scale),
        )

        torch.manual_seed(8)  # to reproduce results
        torch.cuda.manual_seed(1)

    def forward(self, t_embed):
        """
        Args:
            t_embed: (M, self.W) Input Fourier time embeddings
        Returns:
            output: (M, x) Output values
        """
        t_feat = super().forward(t_embed)
        output = self.head(t_feat)
        return output

    def get_vals(self, frame_id=None):
        """Compute values at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            output: (M, x) Output values
        """
        t_embed = self.time_embedding(frame_id)
        output = self.forward(t_embed)
        return output
