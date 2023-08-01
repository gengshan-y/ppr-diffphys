import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import dqtorch

def se3exp_to_vec(se3exp):
    """
    se3exp: B,4,4
    vec: Bx10
    """
    num_bones = se3exp.shape[0]
    device = se3exp.device

    center = se3exp[:,:3,3]
    orient =  se3exp[:,:3,:3]
    orient = dqtorch.matrix_to_quaternion(orient)
    scale = torch.zeros(num_bones,3).to(device)
    vec = torch.cat([center, orient, scale],-1)
    return vec

def vec_to_sim3(vec):
    """
    vec:      ...,10 / ...,8
    center:   ...,3
    orient:   ...,3,3
    scale:    ...,3 / ...,1
    """
    center = vec[...,:3]
    orient = vec[...,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = dqtorch.quaternion_to_matrix(orient) # real first
    scale =  vec[...,7:].exp()
    return center, orient, scale


def rot_angle(mat):
    """
    rotation angle of rotation matrix 
    rmat: ..., 3,3
    """
    eps=1e-4
    cos = (  mat[...,0,0] + mat[...,1,1] + mat[...,2,2] - 1 )/2
    cos = cos.clamp(-1+eps,1-eps)
    angle = torch.acos(cos)
    return angle

def fid_reindex(fid, num_vids, vid_offset):
    """
    re-index absolute frameid {0,....N} to subsets of video id and relative frameid
    fid: N absolution id
    vid: N video id
    tid: N relative id
    """
    tid = torch.zeros_like(fid).float()
    vid = torch.zeros_like(fid)
    max_ts = (vid_offset[1:] - vid_offset[:-1]).max()
    for i in range(num_vids):
        assign = torch.logical_and(fid>=vid_offset[i],
                                    fid<vid_offset[i+1])
        vid[assign] = i
        tid[assign] = fid[assign].float() - vid_offset[i]
        doffset = vid_offset[i+1] - vid_offset[i]
        tid[assign] = (tid[assign] - doffset/2)/max_ts*2
        #tid[assign] = 2*(tid[assign] / doffset)-1
        #tid[assign] = (tid[assign] - doffset/2)/1000.
    return vid, tid

def create_base_se3(bs, device):
    """
    create a base se3 based on near-far plane
    """
    rt = torch.zeros(bs,3,4).to(device)
    rt[:,:3,:3] = torch.eye(3)[None].repeat(bs,1,1).to(device)
    rt[:,:2,3] = 0.
    rt[:,2,3] = 0.3
    return rt

def refine_rt(rt_raw, root_rts):
    """
    input:  rt_raw representing the initial root poses (after scaling)
    input:  root_rts representing delta se3
    output: current estimate of rtks for all frames
    """
    rt_raw = rt_raw.clone()
    root_rmat = root_rts[:,0,:9].view(-1,3,3)
    root_tmat = root_rts[:,0,9:12]

    rmat = rt_raw[:,:3,:3].clone()
    tmat = rt_raw[:,:3,3].clone()
    tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
    rmat = rmat.matmul(root_rmat)
    rt_raw[:,:3,:3] = rmat
    rt_raw[:,:3,3] = tmat
    return rt_raw

def axis_angle_to_matrix(vec):
    quat = dqtorch.axis_angle_to_quaternion(vec)
    mat = dqtorch.quaternion_to_matrix(quat)
    return mat

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    taken from pytorch3d
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    taken from pytorch3d
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def se3_vec2mat(vec):
    """
    torch/numpy function
    vec: ...,7, quaternion real last
    or vec: ...,6, axis angle
    mat: ...,4,4
    """
    shape = vec.shape[:-1]
    if torch.is_tensor(vec):
        mat = torch.zeros(shape+(4,4)).to(vec.device)
        if vec.shape[-1] == 6:
            rmat = axis_angle_to_matrix(vec[...,3:6])
        else:
            vec = vec[...,[0,1,2,6,3,4,5]] # xyzw => wxyz
            rmat = dqtorch.quaternion_to_matrix(vec[...,3:7]) 
        tmat = vec[...,:3]
    else:
        mat = np.zeros(shape+(4,4))
        vec = vec.reshape((-1,vec.shape[-1]))
        if vec.shape[-1]==6:
            rmat = R.from_axis_angle(vec[...,3:6]).as_matrix() # xyzw
        else:
            rmat = R.from_quat(vec[...,3:7]).as_matrix() # xyzw
        tmat = np.asarray(vec[...,:3])
        rmat = rmat.reshape(shape+(3,3))
        tmat = tmat.reshape(shape+(3,))
    mat[...,:3,:3] = rmat
    mat[...,:3,3] = tmat
    mat[...,3,3] = 1
    return mat

def se3_mat2rt(mat):
    """
    numpy function
    mat: ...,4,4
    rmat: ...,3,3
    tmat: ...,3
    """
    rmat = mat[...,:3,:3]
    tmat = mat[...,:3,3]
    return rmat, tmat

def se3_mat2vec(mat, outdim=7):
    """
    mat: ...,4,4
    vec: ...,7
    """
    shape = mat.shape[:-2]
    assert( torch.is_tensor(mat) )
    tmat = mat[...,:3,3]
    quat = dqtorch.matrix_to_quaternion(mat[...,:3,:3]) 
    if outdim==7:
        rot = quat[...,[1,2,3,0]] # xyzw <= wxyz
    elif outdim==6:
        rot = quaternion_to_axis_angle(quat)
    else: print('error'); exit()
    vec = torch.cat([tmat, rot], -1)
    return vec