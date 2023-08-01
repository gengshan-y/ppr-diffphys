import math
import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

def clip_grad(model):
    """
    gradient clipping
    """
    is_invalid_grad=False
    grad_mlp_act=[]
    for name,p in model.named_parameters():
        try: 
            pgrad_nan = p.grad.isnan()
            if pgrad_nan.sum()>0: 
                print(name)
                is_invalid_grad=True
        except: pass
        grad_mlp_act.append(p)
    
    grad_out = clip_grad_norm_(grad_mlp_act,    1)

    if is_invalid_grad:
        zero_grad_list(model.parameters())

    return grad_out

def zero_grad_list(paramlist):
    """
    Clears the gradients of all optimized :class:`torch.Tensor`
    """
    for p in paramlist:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 tscale=1., N_freqs=6,
                 in_channels_xyz=63,
                 out_channels=3, 
                 skips=[4], 
                 activation=nn.ReLU(True), in_channels_code=0):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        skips: add skip connection in the Dth layer
        in_channels_code: only used for nerf_skin,
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_code = in_channels_code
        self.skips = skips
        self.use_xyz = False

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

        # output layers
        self.pred = nn.Linear(W, out_channels)

        self.embed = Embedding(1,N_freqs)
        self.tscale = tscale

    def reinit(self, gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight,'data'):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, inx, xyz=None):
        #TODO
        inx = inx * self.tscale
        #inx = inx * self.tscale * 2 -1
        inx = self.embed(inx)

        xyz_ = inx
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([inx, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        out = self.pred(xyz_)
        return out

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
