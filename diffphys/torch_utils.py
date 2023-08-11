import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F

import sys, os

sys.path.append("%s/../../../../" % os.path.dirname(__file__))
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.nnutils.base import ScaleLayer
from lab4d.nnutils.time import TimeMLP
from lab4d.utils.quat_transform import (
    matrix_to_quaternion,
    quaternion_mul,
    quaternion_translation_to_se3,
)


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


class TimeMLPWrapper(TimeMLP):
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


class CameraMLPWrapper(TimeMLP):
    """Encode camera pose over time (rotation + translation) with an MLP

    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        rtmat,
        frame_info=None,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[1, 2, 3, 4],
        activation=nn.ReLU(True),
    ):
        if frame_info is None:
            num_frames = len(rtmat)
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
        self.trans = nn.Sequential(
            nn.Linear(W, 3),
        )
        self.quat = nn.Sequential(
            nn.Linear(W, 4),
        )

        # camera pose: field to camera
        self.base_quat = nn.Parameter(torch.zeros(self.time_embedding.num_vids, 4))
        self.register_buffer(
            "init_vals", torch.tensor(rtmat, dtype=torch.float32), persistent=False
        )

        # override the loss function
        def loss_fn(gt):
            quat, trans = self.get_vals()
            pred = quaternion_translation_to_se3(quat, trans)
            loss = F.mse_loss(pred, gt)
            # loss_trans = (pred[...,:3,3]-gt[...,:3,3]).norm(2, dim=-1).mean()
            # loss_rot = rot_angle(pred[...,:3,:3]@gt[...,:3,:3].transpose(-1,-2)).mean()
            # loss = (loss_trans + 1.0* loss_rot)
            return loss

        self.loss_fn = loss_fn

    def base_init(self):
        """Initialize base camera rotations from initial camera trajectory"""
        rtmat = self.init_vals
        frame_offset = self.get_frame_offset()
        base_rmat = rtmat[frame_offset[:-1], :3, :3]
        base_quat = matrix_to_quaternion(base_rmat)
        self.base_quat.data = base_quat

    def mlp_init(self):
        """Initialize camera SE(3) transforms from external priors"""
        self.base_init()
        super().mlp_init()
        # super().mlp_init(termination_loss=0.1)

        # with torch.no_grad():
        #     os.makedirs("tmp", exist_ok=True)
        #     draw_cams(rtmat.cpu().numpy()).export("tmp/cameras_gt.obj")
        #     quat, trans = self.get_vals()
        #     rtmat_pred = quaternion_translation_to_se3(quat, trans)
        #     draw_cams(rtmat_pred.cpu()).export("tmp/cameras_pred.obj")

    def forward(self, t_embed):
        """
        Args:
            t_embed: (M, self.W) Input Fourier time embeddings
        Returns:
            quat: (M, 4) Output camera rotation quaternions
            trans: (M, 3) Output camera translations
        """
        t_feat = super().forward(t_embed)
        trans = self.trans(t_feat)
        quat = self.quat(t_feat)
        quat = F.normalize(quat, dim=-1)
        return quat, trans

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        t_embed = self.time_embedding(frame_id)
        quat, trans = self.forward(t_embed)
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id.long()]

        # multiply with per-instance base rotation
        base_quat = self.base_quat[inst_id]
        base_quat = F.normalize(base_quat, dim=-1)
        quat = quaternion_mul(quat, base_quat)
        return quat, trans
