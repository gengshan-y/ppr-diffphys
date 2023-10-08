# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# This is copy-pasted from lab4d. Only called when ppr-diffphys is being used independent of lab4d

import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEmbedding(nn.Module):
    """A Fourier embedding that maps x to (x, sin(2^k x), cos(2^k x), ...)
    Adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py

    Args:
        in_channels (int): Number of input channels (3 for both xyz, direction)
        N_freqs (int): Number of frequency bands
        logscale (bool): If True, construct frequency bands in log-space
        pre_rotate (bool): If True, pre-rotate the input along each plane
    """

    def __init__(self, in_channels, N_freqs, logscale=True, pre_rotate=False):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels

        if pre_rotate:
            # rotate along each dimension for 45 degrees
            rot_mat = get_pre_rotation(in_channels)
            rot_mat = torch.tensor(rot_mat, dtype=torch.float32)
            self.register_buffer("rot_mat", rot_mat, persistent=False)

        # no embedding
        if N_freqs == -1:
            self.out_channels = 0
            return

        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("alpha", torch.tensor(-1.0, dtype=torch.float32))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def set_alpha(self, alpha):
        """Set the alpha parameter for the annealing window

        Args:
            alpha (float): 0 to 1, -1 represents full frequency band
        """
        self.alpha.data = alpha

    def forward(self, x):
        """Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Args:
            x: (B, self.in_channels)
        Returns:
            out: (B, self.out_channels)
        """
        if self.N_freqs == -1:
            return torch.zeros_like(x[..., :0])

        # cosine features
        if self.N_freqs > 0:
            shape = x.shape
            device = x.device
            input_dim = shape[-1]
            output_dim = input_dim * (1 + self.N_freqs * self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)

            # assign input coordinates to the first few output channels
            x = x.reshape(-1, input_dim)
            out = torch.empty(x.shape[0], output_dim, dtype=x.dtype, device=device)
            out[:, :input_dim] = x

            if hasattr(self, "rot_mat"):
                x = x @ self.rot_mat.T
                x = x.view(x.shape[0], input_dim, -1)

            # assign fourier features to the remaining channels
            out_bands = out[:, input_dim:].view(
                -1, self.N_freqs, self.nfuncs, input_dim
            )
            for i, func in enumerate(self.funcs):
                # (B, nfreqs, input_dim) = (1, nfreqs, 1) * (B, 1, input_dim)
                if hasattr(self, "rot_mat"):
                    signal = self.freq_bands[None, :, None, None] * x[:, None]
                    response = func(signal)
                    response = response.view(-1, self.N_freqs, input_dim, x.shape[-1])
                    response = response.mean(-1)
                else:
                    signal = self.freq_bands[None, :, None] * x[:, None, :]
                    response = func(signal)
                out_bands[:, :, i] = response

            self.apply_annealing(out_bands)
            out = out.view(out_shape)
        else:
            out = x
        return out

    def apply_annealing(self, out_bands):
        """Apply the annealing window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )

        Args:
            out_bands: (..., N_freqs, nfuncs, in_channels) Frequency bands
        """
        device = out_bands.device
        if self.alpha >= 0:
            alpha_freq = self.alpha * self.N_freqs
            window = alpha_freq - torch.arange(self.N_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1, -1, 1, 1)
            out_bands[:] = window * out_bands

    def get_mean_embedding(self, device):
        """Compute the mean Fourier embedding

        Args:
            device (torch.device): Output device
        """
        mean_embedding = torch.zeros(self.out_channels, device=device)
        return mean_embedding


class TimeEmbedding(nn.Module):
    """A learnable feature embedding per frame

    Args:
        num_freq_t (int): Number of frequencies in time embedding
        frame_info (Dict): Metadata about the frames in a dataset
        out_channels (int): Number of output channels
    """

    def __init__(self, num_freq_t, frame_info, out_channels=128, time_scale=1.0):
        super().__init__()
        self.fourier_embedding = PosEmbedding(1, num_freq_t)
        t_channels = self.fourier_embedding.out_channels
        self.out_channels = out_channels

        self.frame_offset = frame_info["frame_offset"]
        self.frame_offset_raw = frame_info["frame_offset_raw"]
        self.num_frames = self.frame_offset[-1]
        self.num_vids = len(self.frame_offset) - 1

        frame_mapping = frame_info["frame_mapping"]  # list of list
        frame_mapping = torch.tensor(frame_mapping)  # (M,)
        frame_offset_raw = frame_info["frame_offset_raw"]

        max_ts = (frame_offset_raw[1:] - frame_offset_raw[:-1]).max()
        raw_fid = torch.arange(0, frame_offset_raw[-1])
        raw_fid_to_vid = frameid_to_vid(raw_fid, frame_offset_raw)
        raw_fid_to_vstart = torch.tensor(frame_offset_raw[raw_fid_to_vid])
        raw_fid_to_vidend = torch.tensor(frame_offset_raw[raw_fid_to_vid + 1])
        raw_fid_to_vidlen = raw_fid_to_vidend - raw_fid_to_vstart

        # M
        self.register_buffer(
            "frame_to_vid", raw_fid_to_vid[frame_mapping], persistent=False
        )
        # M, in range [0,N-1], M<N
        self.register_buffer("frame_mapping", frame_mapping, persistent=False)
        frame_mapping_inv = torch.full((frame_mapping.max().item() + 1,), 0)
        frame_mapping_inv[frame_mapping] = torch.arange(len(frame_mapping))
        self.register_buffer("frame_mapping_inv", frame_mapping_inv, persistent=False)
        # N
        self.register_buffer("raw_fid_to_vid", raw_fid_to_vid, persistent=False)
        self.register_buffer("raw_fid_to_vidlen", raw_fid_to_vidlen, persistent=False)
        self.register_buffer("raw_fid_to_vstart", raw_fid_to_vstart, persistent=False)

        # a function, make it more/less senstiive to time
        def frame_to_tid_fn(frame_id):
            if torch.is_tensor(frame_id):
                device = frame_id.device
            else:
                frame_id = torch.tensor(frame_id)
                device = "cpu"
            frame_id = frame_id.to(self.frame_to_vid.device)
            vid_len = self.raw_fid_to_vidlen[frame_id.long()]
            tid_sub = frame_id - self.raw_fid_to_vstart[frame_id.long()]
            tid = (tid_sub - vid_len / 2) / max_ts * 2  # [-1, 1]
            tid = tid * time_scale
            tid = tid.to(device)
            return tid

        self.frame_to_tid = frame_to_tid_fn

        self.inst_embedding = InstEmbedding(self.num_vids, inst_channels=out_channels)
        self.mapping1 = nn.Linear(t_channels, out_channels)
        self.mapping2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, frame_id=None):
        """
        Args:
            frame_id: (...,) Frame id to evaluate at, or None to use all frames
        Returns:
            t_embed (..., self.W): Output time embeddings
        """
        device = self.parameters().__next__().device
        if frame_id is None:
            inst_id, t_sample = self.frame_to_vid, self.frame_to_tid(self.frame_mapping)
        else:
            if not torch.is_tensor(frame_id):
                frame_id = torch.tensor(frame_id, device=device)
            inst_id = self.raw_fid_to_vid[frame_id.long()]
            t_sample = self.frame_to_tid(frame_id)

        if inst_id.ndim == 1:
            inst_id = inst_id[..., None]  # (N, 1)
            t_sample = t_sample[..., None]  # (N, 1)

        coeff = self.fourier_embedding(t_sample)

        inst_code = self.inst_embedding(inst_id[..., 0])
        coeff = self.mapping1(coeff)
        t_embed = torch.cat([coeff, inst_code], -1)
        t_embed = self.mapping2(t_embed)
        return t_embed

    def get_mean_embedding(self, device):
        """Compute the mean time embedding over all frames

        Args:
            device (torch.device): Output device
        """
        t_embed = self.forward(self.frame_mapping).mean(0, keepdim=True)
        # t_embed = self.basis.weight.mean(1)
        return t_embed


class TimeEmbeddingRest(TimeEmbedding):
    """Time embedding with a rest embedding"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rest_embedding = nn.Parameter(torch.zeros(1, self.out_channels))

    def get_mean_embedding(self, device):
        return self.rest_embedding


class InstEmbedding(nn.Module):
    """A learnable embedding per object instance

    Args:
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        inst_channels (int): Number of channels in the instance code
    """

    def __init__(self, num_inst, inst_channels):
        super().__init__()
        self.out_channels = inst_channels
        self.num_inst = num_inst
        self.set_beta_prob(0.0)  # probability of sampling a random instance
        if inst_channels > 0:
            self.mapping = nn.Embedding(num_inst, inst_channels)

    def forward(self, inst_id):
        """
        Args:
            inst_id: (M,) Instance id, or None to use the average instance
        Returns:
            out: (M, self.out_channels)
        """
        if self.out_channels == 0:
            return torch.zeros(inst_id.shape + (0,), device=inst_id.device)
        else:
            if self.num_inst == 1:
                inst_code = self.mapping(torch.zeros_like(inst_id))
            else:
                if self.training and self.beta_prob > 0:
                    inst_id = self.randomize_instance(inst_id)
                inst_code = self.mapping(inst_id)
            return inst_code

    def randomize_instance(self, inst_id):
        """Randomize the instance code with probability beta_prob. Used for
        code swapping regularization

        Args:
            inst_id: (M, ...) Instance id
        Returns:
            inst_id: (M, ...) Randomized instance ids
        """
        minibatch_size = inst_id.shape[0]
        rand_id = torch.randint(self.num_inst, (minibatch_size,), device=inst_id.device)
        rand_id = rand_id.reshape((minibatch_size,) + (1,) * (len(inst_id.shape) - 1))
        rand_id = rand_id.expand_as(inst_id)
        rand_mask = torch.rand_like(rand_id.float()) < self.beta_prob
        inst_id = torch.where(rand_mask, rand_id, inst_id)
        return inst_id

    def get_mean_embedding(self):
        """Compute the mean instance id"""
        return self.mapping.weight.mean(0)

    def set_beta_prob(self, beta_prob):
        """Set the beta parameter for the instance code. This is the probability
        of sampling a random instance code

        Args:
            beta_prob (float): Instance code swapping probability, 0 to 1
        """
        self.beta_prob = beta_prob


class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.FloatTensor([scale]))

    def forward(self, input):
        return input * self.scale


class BaseMLP(nn.Module):
    """Adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py

    Args:
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        final_act (bool): If True, apply the activation function to the output
    """

    def __init__(
        self,
        D=8,
        W=256,
        in_channels=63,
        out_channels=3,
        skips=[4],
        activation=nn.ReLU(True),
        final_act=False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips

        if in_channels == 0:
            return

        # linear layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"linear_{i+1}", layer)
        if final_act:
            self.linear_final = nn.Sequential(nn.Linear(W, out_channels), activation)
        else:
            self.linear_final = nn.Linear(W, out_channels)

    def forward(self, x):
        """
        Args:
            x: (..., self.in_channels)
        Returns:
            out: (..., self.out_channels)
        """
        out = x
        for i in range(self.D):
            if i in self.skips:
                out = torch.cat([x, out], -1)
            out = getattr(self, f"linear_{i+1}")(out)
        out = self.linear_final(out)
        return out


class TimeMLP(BaseMLP):
    """MLP that encodes a quantity over time.

    Args:
        frame_info (Dict): Metadata about the frames in a dataset
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in the time embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        time_scale (float): Control the sensitivity to time by scaling.
            Lower values make the module less sensitive to time.
    """

    def __init__(
        self,
        frame_info,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
        time_scale=1.0,
        bottleneck_dim=None,
        has_rest=False,
    ):
        if bottleneck_dim is None:
            bottleneck_dim = W

        frame_offset = frame_info["frame_offset"]
        # frame_offset_raw = frame_info["frame_offset_raw"]
        if num_freq_t > 0:
            max_ts = (frame_offset[1:] - frame_offset[:-1]).max()
            # scale according to input frequency: num_frames = 64 -> freq = 6
            num_freq_t = np.log2(max_ts / 64) + num_freq_t
            # # scale according to input frequency: num_frames = 512 -> freq = 6
            # num_freq_t = np.log2(max_ts / 512) + num_freq_t
            num_freq_t = int(np.rint(num_freq_t))
            # print("max video len: %d, override num_freq_t to %d" % (max_ts, num_freq_t))

        super().__init__(
            D=D,
            W=W,
            in_channels=bottleneck_dim,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )

        if has_rest:
            arch = TimeEmbeddingRest
        else:
            arch = TimeEmbedding

        self.time_embedding = arch(
            num_freq_t, frame_info, out_channels=bottleneck_dim, time_scale=time_scale
        )

        def loss_fn(y):
            x = self.get_vals()
            return F.mse_loss(x, y)

        self.loss_fn = loss_fn

    def forward(self, t_embed):
        """
        Args:
            t_embed: (..., self.W) Time Fourier embeddings
        Returns:
            out: (..., self.W) Time-dependent features
        """
        t_feat = super().forward(t_embed)
        return t_feat

    def mlp_init(self, loss_fn=None, termination_loss=0.0001):
        """Initialize the time embedding MLP to match external priors.
        `self.init_vals` is defined by the child class, and could be
        (nframes, 4, 4) camera poses or (nframes, 4) camera intrinsics
        """
        if loss_fn is None:
            loss_fn = self.loss_fn

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        i = 0
        while True:
            optimizer.zero_grad()
            loss = loss_fn(self.init_vals)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"iter: {i}, loss: {loss.item():.4f}")
            i += 1
            if loss < termination_loss:
                break

    def compute_distance_to_prior(self):
        """Compute L2-distance from current SE(3) / intrinsics values to
        external priors.

        Returns:
            loss (0,): Mean squared error to priors
        """
        return self.loss_fn(self.init_vals)

    def get_vals(self, frame_id=None):
        """Compute values at the given frames.

        Args:
            frame_id: (...,) Frame id. If None, evaluate at all frames
        Returns:
            pred: Predicted outputs
        """
        t_embed = self.time_embedding(frame_id)
        pred = self.forward(t_embed)
        return pred

    def get_mean_vals(self):
        """Compute the mean embedding over all frames"""
        device = self.parameters().__next__().device
        t_embed = self.time_embedding.get_mean_embedding(device)
        pred = self.forward(t_embed)
        return pred

    def get_frame_offset(self):
        """Return the number of frames before the first frame of each video"""
        return self.time_embedding.frame_offset


def create_plane(size, offset):
    """
    Create a plane mesh spaning x,z axis
    """
    vertices = np.array(
        [
            [-0.5, 0, -0.5],  # vertex 0
            [0.5, 0, -0.5],  # vertex 1
            [0.5, 0, 0.5],  # vertex 2
            [-0.5, 0, 0.5],  # vertex 3
        ]
    )
    vertices = vertices * size + np.asarray(offset)

    faces = np.array(
        [
            [0, 2, 1],  # triangle 0
            [2, 0, 3],  # triangle 1
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def create_floor_mesh(scale=20, gl=True):
    # create scene
    floor1 = create_plane(scale, [0, 0, 0])
    floor1.visual.vertex_colors[:, 0] = 10
    floor1.visual.vertex_colors[:, 1] = 255
    floor1.visual.vertex_colors[:, 2] = 102
    floor1.visual.vertex_colors[:, 3] = 102

    floor2 = create_plane(scale / 4, [0, 0.01, 0])
    floor2.visual.vertex_colors[:, 0] = 10
    floor2.visual.vertex_colors[:, 1] = 102
    floor2.visual.vertex_colors[:, 2] = 255
    floor2.visual.vertex_colors[:, 3] = 102

    floor = trimesh.util.concatenate([floor1, floor2])
    if not gl:
        floor.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return floor


def frameid_to_vid(fid, frame_offset):
    """Given absolute frame ids [0, ..., N], compute the video id of each frame.

    Args:
        fid: (nframes,) Absolute frame ids
          e.g. [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203]
        frame_offset: (nvideos + 1,) Offset of each video
          e.g., [0, 100, 200, 300]
    Returns:
        vid: (nframes,) Maps idx to video id
        tid: (nframes,) Maps idx to relative frame id
    """
    vid = torch.zeros_like(fid)
    for i in range(frame_offset.shape[0] - 1):
        assign = torch.logical_and(fid >= frame_offset[i], fid < frame_offset[i + 1])
        vid[assign] = i
    return vid


def match_param_name(name, param_lr, type):
    """
    Match the param name with the param_lr dict

    Args:
        name (str): the name of the param
        param_lr (Dict): the param_lr dict
        type (str): "with" or "startwith"

    Returns:
        bool, lr
    """
    matched_param = []
    matched_lr = []

    for params_name, lr in param_lr.items():
        if type == "with":
            if params_name in name:
                matched_param.append(params_name)
                matched_lr.append(lr)
        elif type == "startwith":
            if name.startswith(params_name):
                matched_param.append(params_name)
                matched_lr.append(lr)
        else:
            raise ValueError("type not found")

    if len(matched_param) == 0:
        return False, 0.0
    elif len(matched_param) == 1:
        return True, matched_lr[0]
    else:
        raise ValueError("multiple matches found", matched_param)


def interp_wt(x, y, x2, type="linear"):
    """Map a scalar value from range [x0, x1] to [y0, y1] using interpolation

    Args:
        x: Input range [x0, x1]
        y: Output range [y0, y1]
        x2 (float): Scalar value in range [x0, x1]
        type (str): Interpolation type ("linear" or "log")
    Returns:
        y2 (float): Scalar value mapped to [y0, y1]
    """
    # Extract values from tuples
    x0, x1 = x
    y0, y1 = y

    # # Check if x2 is in range
    # if x2 < x0 or x2 > x1:
    #     raise ValueError("x2 must be in the range [x0, x1]")

    if type == "linear":
        # Perform linear interpolation
        y2 = y0 + (x2 - x0) * (y1 - y0) / (x1 - x0)

    elif type == "log":
        # Transform to log space
        log_y0 = np.log10(y0)
        log_y1 = np.log10(y1)

        # Perform linear interpolation in log space
        log_y2 = log_y0 + (x2 - x0) * (log_y1 - log_y0) / (x1 - x0)

        # Transform back to original space
        y2 = 10**log_y2
    elif type == "exp":
        # clip
        assert x0 >= 1
        assert x1 >= 1
        x2 = np.clip(x2, x0, x1)
        # Transform to log space
        log_x0 = np.log10(x0)
        log_x1 = np.log10(x1)
        log_x2 = np.log10(x2)

        # Perform linear interpolation in log space
        y2 = y0 + (log_x2 - log_x0) * (y1 - y0) / (log_x1 - log_x0)
    else:
        raise ValueError("interpolation_type must be 'linear' or 'log'")

    y2 = np.clip(y2, np.min(y), np.max(y))
    return y2
