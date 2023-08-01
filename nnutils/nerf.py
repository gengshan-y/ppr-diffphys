# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import pdb
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from pytorch3d import transforms
import trimesh
from nnutils.geom_utils import fid_reindex
from nnutils.urdf_utils import angle_to_rts
from nnutils.rendering import render_rays
from nnutils.geom_utils import raycast, near_far_to_bound, chunk_rays, \
                            create_base_se3, refine_rt
from collections import defaultdict

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True, alpha=None):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        if alpha is None:
            self.alpha = self.N_freqs
        else: self.alpha = alpha

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        # consine features
        if self.N_freqs>0:
            shape = x.shape
            bs = shape[0]
            input_dim = shape[-1]
            output_dim = input_dim*(1+self.N_freqs*self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)
            device = x.device

            x = x.view(-1,input_dim)
            out = []
            for freq in self.freq_bands:
                for func in self.funcs:
                    out += [func(freq*x)]
            out =  torch.cat(out, -1)

            ## Apply the window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )
            out = out.view(-1, self.N_freqs, self.nfuncs, input_dim)
            window = self.alpha - torch.arange(self.N_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1,-1, 1, 1)
            out = window * out
            out = out.view(-1,self.N_freqs*self.nfuncs*input_dim)

            out = torch.cat([x, out],-1)
            out = out.view(out_shape)
        else: out = x
        return out



class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 out_channels=3, 
                 skips=[4], raw_feat=False, init_beta=1./100, 
                 activation=nn.ReLU(True), in_channels_code=0, vid_code=None):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        in_channels_code: only used for nerf_skin,
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_code = in_channels_code
        self.skips = skips
        self.use_xyz = False

        # video code
        self.vid_code = vid_code
        if vid_code is not None:
            self.num_vid, self.num_codedim = self.vid_code.weight.shape
            in_channels_xyz += self.num_codedim
            self.rand_ratio = 1. # 1: fully random

        # xyz encoding layers
        self.weights_reg = []
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i+1}")
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i+1}")
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                activation)

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, out_channels),
                        )

        self.raw_feat = raw_feat

        self.beta = torch.Tensor([init_beta]) # logbeta
        self.beta = nn.Parameter(self.beta)
        self.symm_ratio = 0
        self.rand_ratio = 0
        self.use_dir = False

#        for m in self.modules():
#            if isinstance(m, nn.Linear):
#                if hasattr(m.weight,'data'):
#                    nn.init.xavier_uniform_(m.weight)

    def reinit(self,gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x ,xyz=None, vidid=None, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
            raw_feat: does not apply sigmoid
            vidid: same size as input_xyz

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, 0], dim=-1)

        # add instance shape
        if self.vid_code is not None: 
            if vidid is None:
                vid_code = self.vid_code.weight.mean(0).expand(input_xyz.shape[:-1] + (-1,))
            else:
                vid_code = self.vid_code(vidid)
            if self.training:
                ##TODO 
                vidid = torch.randint(self.num_vid, input_xyz.shape[:1])
                vidid = vidid.to(input_xyz.device)
                rand_code = self.vid_code(vidid)
                rand_code = rand_code[:,None].expand(vid_code.shape)
                rand_mask = torch.rand_like(vidid.float()) < self.rand_ratio
                vid_code = torch.where(rand_mask[:,None,None], rand_code, vid_code)
            input_xyz = torch.cat([input_xyz, vid_code],-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        if self.raw_feat:
            out = rgb
        else:
            rgb = rgb.sigmoid()
            out = torch.cat([rgb, sigma], -1)
        return out

class Transhead(NeRF):
    """
    translation head
    """
    def __init__(self, **kwargs):
        super(Transhead, self).__init__(**kwargs)

    def forward(self, x, xyz=None,sigma_only=False):
        flow = super(Transhead, self).forward(x, sigma_only=sigma_only)
        flow = flow*0.1
        return flow

class SE3head(NeRF):
    """
    modify the output to be rigid transforms per point
    modified from Nerfies
    """
    def __init__(self, **kwargs):
        super(SE3head, self).__init__(**kwargs)
        self.use_xyz=True

    def forward(self, x, xyz=None,sigma_only=False):
        x = super(SE3head, self).forward(x, sigma_only=sigma_only)
        x = x.view(-1,9)
        rotation, pivot, translation = x.split([3,3,3],-1)
        pivot = pivot*0.1
        translation = translation*0.1
        
        shape = xyz.shape
        warped_points = xyz.view(-1,3).clone()
        warped_points = warped_points + pivot
        rotmat = transforms.so3_exponential_map(rotation)
        warped_points = rotmat.matmul(warped_points[...,None])[...,0]
        warped_points = warped_points - pivot
        warped_points = warped_points + translation

        flow = warped_points.view(shape) - xyz
        return flow

class RTHead(NeRF):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, use_quat, **kwargs):
        super(RTHead, self).__init__(**kwargs)
        # use quaternion when estimating full rotation
        # use exponential map when estimating delta rotation
        self.use_quat=use_quat
        if self.use_quat: self.num_output=7
        else: self.num_output=6
        self.scale_t = 0.1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
        self.reinit(gain=1)

    def reinit(self, gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(RTHead, self).forward(x)
        bs = x.shape[0]
        rts = x.view(-1,self.num_output)  # bs B,x
        B = rts.shape[0]//bs

        tmat= rts[:,0:3] * self.scale_t

        if self.use_quat:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        else:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts

class TrnHeadIntr(NeRF):
    """
    translation with intrinsics
    """
    def __init__(self, **kwargs):
        super(TrnHeadIntr, self).__init__(**kwargs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
        self.reinit(gain=1)

    def reinit(self, gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x, kvec_embedded):
        #x = kvec_embedded
        x = torch.cat([x, kvec_embedded],-1)
        x = super(TrnHeadIntr, self).forward(x)
        tvec = x.view(-1,1,3)
        return tvec
            
class SkelHead(NeRF):
    """
    modify the output to be rigid transforms from a kinematic chain
    """
    def __init__(self, urdf, joints, sim3, rest_angles, **kwargs):
        super(SkelHead, self).__init__(**kwargs)
        self.urdf = urdf
        self.sim3 = nn.Parameter(sim3)
        self.joints = joints
        #self.joints = nn.Parameter(joints)
        jlen_scale = torch.zeros(len(joints))
        self.jlen_scale = nn.Parameter(jlen_scale) # optimize bone length
        self.rest_angles = rest_angles

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        # delta se3 from zero pose
        # output: NxBx(9 rotation + 3 translation)
        bs = x.shape[0]
        device = x.device

        x = super(SkelHead, self).forward(x)
        angles = x.view(bs,-1)  # bs B-1: 0,1,2; 3,4,5; ...
        ##TODO debug
        #if angles.shape[0]>1:
        #    angles[:]=0
        #    angles[:,1] = -1
        angles = angles + self.rest_angles.to(device)
        angles_z = torch.zeros_like(angles) + self.rest_angles.to(device)

        # convert from angles to rts
        joints = self.update_joints(self.urdf, self.joints, self.jlen_scale)
        fk   = angle_to_rts(self.urdf, joints, angles, self.sim3) # bs,B,4,4
        fk_z = angle_to_rts(self.urdf, joints, angles_z, self.sim3) # bs,B,4,4
        rmat = fk[...,:3,:3]
        rmat_z=fk_z[...,:3,:3]
        tmat = fk[...,:3,3]
        tmat_z=fk_z[...,:3,3]

        rmat_zi = rmat_z.permute(0,1,3,2)
        tmat_zi = -rmat_z.permute(0,1,3,2).matmul(tmat_z[...,None])[...,0]

        # world points transforms from zero to posed
        tmat = rmat.matmul(tmat_zi[...,None])[...,0] + tmat
        rmat = rmat.matmul(rmat_zi)

        rmat = rmat.reshape(-1,9)
        tmat = tmat.reshape(-1,3)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts    

    def forward_abs(self, x):
        # absolute se3
        # points: joint coordinate to root coordinate
        # output: NxBx(9 rotation + 3 translation)
        bs = x.shape[0]
        device = x.device

        x = super(SkelHead, self).forward(x)
        angles = x.view(bs,-1)  # bs B
        angles = angles + self.rest_angles.to(device)

        # convert from angles to rts
        joints = self.update_joints(self.urdf, self.joints, self.jlen_scale)
        fk = angle_to_rts(self.urdf, joints, angles, self.sim3) # bs,B,4,4
        rmat = fk[...,:3,:3]
        tmat = fk[...,:3,3]

        rmat = rmat.reshape(-1,9)
        tmat = tmat.reshape(-1,3)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts, angles

    @staticmethod
    def update_joints(urdf, joints, jlen_scale):
        """
        scale bone length
        joints, N,3
        jlen_scale, N log scales
        """
        joints = joints.clone()
        jlen_scale = jlen_scale.clone()
        if urdf.robot_name=='a1':
            jlen_scale = jlen_scale.view(2,2,3)
            jlen_scale = jlen_scale.mean(1)[:,None].repeat(1,2,1)
            jlen_scale = jlen_scale.view(-1)
        joints = joints * jlen_scale.exp()[:,None]
        return joints


class FrameCode(nn.Module):
    """
    frame index and video index to code
    """
    def __init__(self, num_freq, embedding_dim, vid_offset, scale=1):
        super(FrameCode, self).__init__()
        self.vid_offset = vid_offset
        self.num_vids = len(vid_offset)-1
        # compute maximum frequency:64-127 frame=>10
        max_ts = (self.vid_offset[1:] - self.vid_offset[:-1]).max()
        if num_freq>0:
            self.num_freq = 2*int(np.log2(max_ts))-2
        else:
            self.num_freq = 0
        #self.num_freq = num_freq

        self.fourier_embed = Embedding(1,self.num_freq,alpha=self.num_freq)
        #self.fourier_embed = Embedding(1,num_freq,alpha=num_freq)
        self.basis_mlp = nn.Linear(self.num_vids*self.fourier_embed.out_channels,
                                embedding_dim)
        self.scale = scale # input scale factor
        self.reinit(gain=1)

    def reinit(self,gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, fid):
        """
        fid->code: N->N,embedding_dim
        """
        bs = fid.shape[0]
        vid, tid = fid_reindex(fid, self.num_vids, self.vid_offset)
        tid = tid*self.scale
        tid = tid.view(bs,1)
        vid = vid.view(bs,1).long()
        coeff = self.fourier_embed(tid) # N, n_channels
        vid = F.one_hot(vid, num_classes=self.num_vids) # N, 1, num_vids
        # pad zeros for each
        coeff = coeff[...,None] * vid # N, n_channels, num_vids
        coeff = coeff.view(bs, -1)
        code = self.basis_mlp(coeff)
        return code

class RTExplicit(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, delta=False, rand=True):
        super(RTExplicit, self).__init__()
        self.max_t = max_t
        self.delta = delta

        # initialize rotation
        trans = torch.zeros(max_t, 3)
        if delta:
            rot = torch.zeros(max_t, 3) 
        else:
            if rand:
                rot = torch.rand(max_t, 4) * 2 - 1
            else:
                rot = torch.zeros(max_t, 4)
                rot[:,0] = 1
        se3 = torch.cat([trans, rot],-1)

        self.se3 = nn.Parameter(se3)
        self.num_output = se3.shape[-1]


    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        bs = x.shape[0]
        x = self.se3[x.long()] # bs B,x
        rts = x.view(-1,self.num_output)
        B = rts.shape[0]//bs
        
        tmat= rts[:,0:3] *0.1

        if self.delta:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        else:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts

class RTExpMLP(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, num_freqs, t_embed_dim, data_offset, delta=False):
        super(RTExpMLP, self).__init__()
        #self.root_code = nn.Embedding(max_t, t_embed_dim)
        self.root_code = FrameCode(num_freqs, t_embed_dim, data_offset, scale=0.1)
        #self.root_code = FrameCode(num_freqs, t_embed_dim, data_offset)

        self.base_rt = RTExplicit(max_t, delta=delta,rand=False)
        #self.base_rt = RTHead(use_quat=True, 
        #            D=2, W=64,
        #            in_channels_xyz=t_embed_dim,in_channels_dir=0,
        #            out_channels=7, raw_feat=True)
        #self.base_rt = nn.Sequential(self.root_code, self.base_rt)
        self.mlp_rt = RTHead(use_quat=False, 
                    in_channels_xyz=t_embed_dim,in_channels_dir=0,
                    out_channels=6, raw_feat=True)
        self.delta_rt = nn.Sequential(self.root_code, self.mlp_rt)

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        base_rts = self.base_rt(x)
        delt_rts = self.delta_rt(x)

        # magnify gradient by 10x
        base_rts = base_rts * 10 - (base_rts*9).detach()
        
        rmat = base_rts[:,0,:9].view(-1,3,3)
        tmat = base_rts[:,0,9:12]
        
        delt_rmat = delt_rts[:,0,:9].view(-1,3,3)
        delt_tmat = delt_rts[:,0,9:12]
    
        tmat = tmat + rmat.matmul(delt_tmat[...,None])[...,0]
        #tmat = tmat + delt_tmat
        rmat = rmat.matmul(delt_rmat)
        
        rmat = rmat.view(-1,9)
        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(-1,1,12)
        return rts

class ScoreHead(NeRF):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, recursion_level, **kwargs):
        super(ScoreHead, self).__init__(**kwargs)
        grid= generate_healpix_grid(recursion_level=recursion_level)
        self.register_buffer('grid', grid)
        self.num_scores = self.grid.shape[0]

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(ScoreHead, self).forward(x)
        bs = x.shape[0]
        x = x.view(-1,self.num_scores+3)  # bs B,x

        # do not use tmat since it is not trained
        tmat = x[:,0:3]*0.
        scores=x[:,3:]
        if self.training:
            return scores, self.grid
        else:
            scores = scores.view(bs,-1,1)
            rmat = self.grid[None].repeat(bs,1,1,1)
            tmat = tmat[:,None].repeat(1,self.num_scores,1)
            rmat = rmat.view(bs,-1,9)
            rts = torch.cat([scores,rmat, tmat],-1)
            rts = rts.view(bs,self.num_scores,-1)
            return rts

class NeRFFeat(NeRF):
    """
    nerf feature
    """
    def __init__(self, **kwargs):
        super(NeRFFeat, self).__init__(**kwargs)
        self.conv3d_net = nn.Sequential(
           Conv3dBlock(4,2,stride=(1,1,1)),
                Conv3d(2,1,3, (1,1,1),1,bias=True),
                )
        self.conv3d_refine = nn.Sequential(
           Conv3dBlock(4,2,stride=(1,1,1)),
                Conv3d(2,1,3, (1,1,1),1,bias=True),
                )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x, xyz=None,sigma_only=False):
        feat = super(NeRFFeat, self).forward(x)
        return feat

class Conv3dBlock(nn.Module):
    '''
    3d convolution block as 2 convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1)):
        super(Conv3dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes,stride)
        self.conv1 = Conv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = Conv3d(out_planes, out_planes, 3, (1,1,1), 1)
            

    def forward(self,x):
        out = F.relu(self.conv1(x),inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out),inplace=True)
        return out

def Conv3d(in_planes, out_planes, kernel_size, stride, pad,bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                         nn.BatchNorm3d(out_planes))

class projfeat3d(nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1,1), padding=(0,0), stride=stride[:2],bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self,x):
        b,c,d,h,w = x.size()
        x = self.conv1(x.view(b,c,d,h*w))
        x = self.bn(x)
        x = x.view(b,-1,d//self.stride[0],h,w)
        return x

class NeRFUnc(NeRF):
    """
    nerf uncertainty
    """
    def __init__(self, **kwargs):
        super(NeRFUnc, self).__init__(**kwargs)

    def forward(self, x, xyz=None,sigma_only=False):
        unc = super(NeRFUnc, self).forward(x, sigma_only=sigma_only)
        return unc

class NeRFBG(nn.Module):
    """
    nerf background
    """
    def __init__(self, num_freqs, data_offset, config, opts,**kwargs):
        super(NeRFBG, self).__init__()
        self.nerf_mlp = NeRF(**kwargs)
        self.nerf_mlp.use_dir = True
        self.embedding_xyz = Embedding(3, num_freqs)
        self.embedding_dir = Embedding(3,4 )

        # pose mlp
        self.num_fr = data_offset[-1]
        self.num_vid = len(data_offset)-1
        self.data_offset = data_offset
        t_embed_dim = 128
        self.cam_mlp = RTExpMLP(self.num_fr, num_freqs, t_embed_dim, data_offset)
       
        # nf plane 
        self.near_far = torch.zeros(self.num_fr,2)
        self.near_far = nn.Parameter(self.near_far)
            
        ## near far planes and cams
        # from utils.io import load_root
        #near_far = []
        #rmat = []
        #tmat = []
        #for i in range(self.num_vid):
        #    nf = np.loadtxt('%s/near_far.txt'%\
        #                config.get('data_%d'%i, 'rtk_path_bg'))
        #    near_far.append(nf)
        #    cams = load_root('%s/cam'%config.get('data_%d'%i, 'rtk_path_bg'),-1)
        #    cams = np.stack(cams,0)
        #    rmat.append(cams[:,:3,:3])
        #    tmat.append(cams[:,:3,3])
        #
        #rmat = torch.Tensor(np.concatenate(rmat,0))
        #quat = transforms.matrix_to_quaternion(rmat)
        #tmat = torch.Tensor(np.concatenate(tmat,0))
        #self.cam_mlp.base_rt.se3.data = torch.cat([tmat, quat],-1)
        #near_far = np.concatenate(near_far,0) * 0.1 # downscale
        #near_far = torch.Tensor(near_far)
        #self.near_far.data = near_far

        # others
        self.obj_bound = near_far_to_bound(self.near_far)
        self.ndepth = 128
        self.progress = 0
        self.opts = opts
        self.nerf_models = {'coarse': self.nerf_mlp}
        self.embeddings = {'xyz':self.embedding_xyz, 'dir':self.embedding_dir}

        # load weights
        states = torch.load('tmp/bg.pth', map_location='cpu')
        #states = torch.load('%s/bg.pth'%config.get('data_%d'%i, 'rtk_path_bg'), 
        #           map_location='cpu')
        nerf_states = self.rm_module_prefix(states, 
                    prefix='module.nerf_coarse')
        cam_states = self.rm_module_prefix(states, 
                    prefix='module.nerf_root_rts')
        self.nerf_mlp.load_state_dict(nerf_states, strict=False)
        self.cam_mlp.load_state_dict(cam_states, strict=False)
        # load near far
        self.near_far.data = states['module.near_far']

    @staticmethod
    def rm_module_prefix(states, prefix='module'):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[:len(prefix)] == prefix:
                i = i[len(prefix)+1:]
                new_dict[i] = v
        return new_dict

    def forward(self, xy, t, auto_reshape=True):
        # compute cams
        bs = t.shape[0]
        device = t.device
        rts = self.cam_mlp(t)
        rts = rts.detach() # TODO
        rts_base = create_base_se3(bs, device)
        rts = refine_rt(rts_base, rts)
        Rmat = rts[:,:3,:3]
        Tmat = rts[:,:3,3]
        Kinv = torch.eye(3,device=device)[None].repeat(bs,1,1)
        
        # raycast
        near_far = self.near_far[t]
        rays = raycast(xy, Rmat, Tmat, Kinv, near_far)
        vidid,_ = fid_reindex(t, self.num_vid, self.data_offset)
        rays['vidid'] = vidid[:,None,None].repeat(1,rays['nsample'],1)
        del rays['rtk_vec']
        del rays['xys']
        del rays['xy_uncrop']

        # query mlp
        bs_rays = rays['bs'] * rays['nsample'] # over pixels
        opts = self.opts
        results=defaultdict(list)
        for i in range(0, bs_rays, opts.chunk):
            rays_chunk = chunk_rays(rays,i,opts.chunk)
            # decide whether to use fine samples 
            if self.progress > opts.fine_steps:
                use_fine = True
            else:
                use_fine = False
            rendered_chunks = render_rays(self.nerf_models,
                        self.embeddings,
                        rays_chunk,
                        N_samples = self.ndepth,
                        use_disp=False,
                        perturb=False,
                        noise_std=0,
                        chunk=2048, # chunk size is effective in val mode
                        obj_bound=self.obj_bound,
                        use_fine=use_fine,
                        img_size=-1, # not used
                        progress=self.progress,
                        opts=opts,
                        )
            for k, v in rendered_chunks.items():
                results[k] += [v]
        for k, v in results.items():
            if v[0].dim()==0: # loss
                v = torch.stack(v).mean()
            else:
                if isinstance(v, list):
                    v = torch.cat(v, 0)
                if self.training or not auto_reshape:
                    v = v.view(rays['bs'],rays['nsample'],-1)
                else:
                    v = v.view(bs,opts.render_size, opts.render_size, -1)
            results[k] = v
        return results['img_coarse'], results['sil_coarse'], results['depth_rnd']

class NeRFTransient(NeRF):
    """
    nerf transient
    """
    def __init__(self, num_freqs, tcode_dim, data_offset, **kwargs):
        super(NeRFTransient, self).__init__(**kwargs)
        self.xyembed = Embedding(2,num_freqs)
        self.tcode = FrameCode(num_freqs, tcode_dim, data_offset)

    def forward(self, xy, t):
        xy = xy.clone()
        t = t.clone()
        #xy[...,0] = xy[...,0] / xy[...,0].max() * 2 -1
        #xy[...,1] = xy[...,1] / xy[...,1].max() * 2 -1
        xy_embedded = self.xyembed(xy)
        t_embedded = self.tcode(t)[:,None].repeat(1,xy.shape[1],1)
        xyt_code = torch.cat([xy_embedded, t_embedded],-1)

        rgb = super(NeRFTransient, self).forward(xyt_code, sigma_only=False)
        #rgb = rgb.sigmoid()
        return rgb

class ResNetConv(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    def __init__(self, in_channels):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        if in_channels!=3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc=None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class ConvProj(nn.Module):
    def __init__(self, num_feat):
        super(ConvProj, self).__init__()
        n_hidden = 128
        self.proj1 = conv2d(True, num_feat, n_hidden, kernel_size=1)
        self.proj2 = conv2d(True, n_hidden, n_hidden, kernel_size=1)
        self.proj3 = nn.Conv2d(n_hidden, num_feat, 
                        kernel_size=1, stride=1, padding=0, bias=True)
        self.proj = nn.Sequential(self.proj1,self.proj2, 
                                 self.proj3)

    def forward(self, feat):
        feat = self.proj(feat)
        return feat

class Encoder(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, in_channels=3,out_channels=128, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(in_channels=in_channels)
        self.conv1 = conv2d(batch_norm, 512, out_channels, stride=1, kernel_size=3)
        #net_init(self.conv1)

    def forward(self, img):
        feat = self.resnet_conv.forward(img) # 512,4,4
        feat = self.conv1(feat) # 128,4,4
        feat = F.max_pool2d(feat, 4, 4)
        feat = feat.view(img.size(0), -1)
        return feat

## 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )

def grab_xyz_weights(nerf_model, clone=False):
    """
    zero grad for coarse component connected to inputs, 
    and return intermediate params
    """
    param_list = []
    input_layers=[0]+nerf_model.skips

    input_wt_names = []
    for layer in input_layers:
        input_wt_names.append(f"xyz_encoding_{layer+1}.0.weight")

    for name,p in nerf_model.named_parameters():
        if name in input_wt_names:
            # equiv since the wt after pos_dim does not change
            if clone:
                param_list.append(p.detach().clone()) 
            else:
                param_list.append(p) 
            ## get the weights according to coarse posec
            ## 63 = 3 + 60
            ## 60 = (num_freqs, 2, 3)
            #out_dim = p.shape[0]
            #pos_dim = nerf_model.in_channels_xyz-nerf_model.in_channels_code
            #param_list.append(p[:,:pos_dim]) # 
    return param_list

class BANMoCNN(nn.Module):
    def __init__(self, cnn_in_channels):
        super(BANMoCNN, self).__init__()
        self.encoder = Encoder((112,112), in_channels=cnn_in_channels,
                        out_channels=32)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.decoder_rot = RTHead(use_quat=True, D=1,
                    in_channels_xyz=32*4,in_channels_dir=0,
                    #in_channels_xyz=32,in_channels_dir=0,
                    out_channels=7, raw_feat=True)
        
        #self.embedding_kvec = Embedding(4, 6)
        #in_channels_xyz=4+4*6*2
        #self.decoder_trn = TrnHeadIntr(D=4,
        #            in_channels_xyz=4*32+in_channels_xyz,in_channels_dir=0,
        #            #in_channels_xyz=32+in_channels_xyz,in_channels_dir=0,
        #            out_channels=3, raw_feat=True)
        
        #from nnutils.mono import mono 
        #self.decoder_depth = mono()
        #self.decoder_depth = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        #self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform


    def forward(self, imgs):
        #dp_feats = F.interpolate(dp_feats, (112,112), mode='bilinear')
        #feats = self.encoder(dp_feats)
        imgs = F.interpolate(imgs, (256,256), mode='bilinear')
        imgs_prsd = torch.stack([self.resnet_transform(x) for x in imgs])
        feats = self.encoder(imgs_prsd)
        rot_pred = self.decoder_rot(feats)
        rot_pred[...,9:] = 0 # zero translation

        # translation branch
        #kvec_embedded = self.embedding_kvec(kvec)
        #trn_pred = self.decoder_trn(feats, kvec_embedded)
        #root_pred = torch.cat([rot_pred, trn_pred],-1)

        # depth branch
        #imgs = self.midas_transform(imgs)
        #depth = 1. / self.decoder_depth(imgs)
        #depth = self.decoder_depth(imgs)
        return rot_pred

