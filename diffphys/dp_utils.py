import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import dqtorch

from diffphys.geom_utils import (
    se3_vec2mat,
    se3_mat2vec,
    rot_angle,
    axis_angle_to_matrix,
    quaternion_invert,
)
from diffphys.urdf_utils import (
    articulate_robot_rbrt,
    articulate_robot,
)


def compose_delta(target_q, delta_root):
    """
    target_q: bs, T, 7
    delta_root: bs, T, 6
    """
    delta_qmat = se3_vec2mat(delta_root)
    target_qmat = se3_vec2mat(target_q)
    target_qmat = delta_qmat @ target_qmat
    target_q = se3_mat2vec(target_qmat)
    return target_q


def zero_grad_list(paramlist):
    """
    Clears the gradients of all optimized :class:`torch.Tensor`
    """
    for p in paramlist:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def remove_nan(q_init_grad, bs):
    """
    q_init_grad: bs*xxx
    """
    # original_shape = q_init_grad.shape
    # q_init_grad = q_init_grad.view(bs,-1)
    # invalid_batch = q_init_grad.isnan().sum(1).bool()
    # q_init_grad[invalid_batch] = 0
    # q_init_grad = q_init_grad.reshape(original_shape)
    # clip grad
    q_init_grad[q_init_grad.isnan()] = 0
    clip_th = 0.01
    q_init_grad[q_init_grad > clip_th] = clip_th
    q_init_grad[q_init_grad < -clip_th] = -clip_th
    if q_init_grad.isnan().sum() > 0:
        pdb.set_trace()


def rotate_frame(global_q, target_q):
    """
    global_q: 7
    target_q: bs,t,7
    """
    # root states: T = Tg @ T_t, bs,T
    global_qmat = se3_vec2mat(global_q)
    if len(global_q.shape) == 1:
        global_qmat = global_qmat[None, None]
    target_qmat = se3_vec2mat(target_q)
    target_qmat = global_qmat @ target_qmat
    target_q = se3_mat2vec(target_qmat, outdim=target_q.shape[-1])
    return target_q


def rotate_frame_vel(global_q, target_qd):
    # only rotate the first 3 elements
    global_qd = global_q.clone()
    global_qd[..., :3] = 0
    target_qd_rev = torch.cat([target_qd[..., 3:], target_qd[..., :3]], -1)
    rot = rotate_frame(global_qd, target_qd)[..., :3]
    trn = rotate_frame(global_qd, target_qd_rev)[..., :3]
    target_qd_rt = torch.cat([rot, trn], -1)
    return target_qd_rt


def compute_com(body_q, part_com, part_mass):
    body_com = R.from_quat(body_q[:, 3:]).as_matrix() @ part_com
    body_com = body_com[..., 0] + body_q[:, :3]
    com = (body_com * part_mass[:, None]).sum(0) / part_mass.sum()
    return com


def reduce_loss(loss_seq, clip=False, th=0):
    """
    bs,T
    """
    if clip:
        for i in range(len(loss_seq)):
            if th == 0:
                loss_sub = loss_seq[i]
                th = loss_sub[loss_sub > 0].median() * 10
            clip_val, clip_idx = torch.max(loss_seq[i] > th, 0)
            if clip_val == 1:
                loss_seq[i, clip_idx:] = 0
                print("clipped env %d at %d" % (i, clip_idx))
    if loss_seq.sum() > 0:
        loss_seq = loss_seq[loss_seq > 0].mean()
    else:
        loss_seq = loss_seq.mean()
    return loss_seq


def se3_loss(pred, gt, rot_ratio=0.1):
    """
    ...,7
    """
    # find nan values
    nanid = torch.logical_or(pred.sum(-1).isnan(), gt.sum(-1).isnan())

    trn_loss = (pred[..., :3] - gt[..., :3]).pow(2).sum(-1)

    rot_pred = pred[..., 3:]
    rot_gt = gt[..., 3:]
    if rot_pred.shape[-1] == 3:
        rot_pred = axis_angle_to_matrix(rot_pred)
        rot_gt = axis_angle_to_matrix(rot_gt)
        rot_gti = rot_gt.transpose(-1, -2)
    elif rot_pred.shape[-1] == 4:
        rot_pred = dqtorch.quaternion_to_matrix(
            rot_pred[..., [3, 0, 1, 2]]
        )  # xyzw => wxyz
        rot_gti = quaternion_invert(rot_gt[..., [3, 0, 1, 2]])
        rot_gti = dqtorch.quaternion_to_matrix(rot_gti)
    rot_loss = rot_angle(rot_pred @ rot_gti)

    loss = trn_loss + rot_loss * rot_ratio
    loss[nanid] = 0
    return loss


def bullet2gl(msm, in_bullet):
    # in_bullet: convert the rest mesh as well (for those in bullet)
    ndim = msm["pos"].ndim - 1
    issac_to_gl = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).reshape(
        ndim * (1,) + (3, 3)
    )
    msm["pos"] = (issac_to_gl @ msm["pos"][..., None])[..., 0]
    if in_bullet:
        shape = msm["orn"].shape[:-1]
        orn = R.from_quat(msm["orn"].reshape((-1, 4))).as_matrix()  # N,3,3
        msm["orn"] = R.from_matrix(orn @ issac_to_gl.reshape((-1, 3, 3))).as_quat()
        msm["orn"] = msm["orn"].reshape(shape + (4,))
    msm["orn"][..., :3] = (issac_to_gl @ msm["orn"][..., :3, None])[..., 0]  # xyzw

    msm["vel"] = (issac_to_gl @ msm["vel"][..., None])[..., 0]
    msm["avel"] = (issac_to_gl @ msm["avel"][..., None])[..., 0]


def can2gym2gl(
    x_rest, obs, gforce=None, com=None, in_bullet=False, use_urdf=False, use_angle=False
):
    if use_urdf:
        if use_angle:
            cfg = np.asarray(obs["jang"])
            mesh = articulate_robot(x_rest, cfg=cfg, use_collision=True)
            rmat = R.from_quat(obs["orn"]).as_matrix()  # xyzw
            tmat = np.asarray(obs["pos"])
            mesh.vertices = mesh.vertices @ rmat.T + tmat[None]
        else:
            # need to parse sperical joints => assuming it's going over joints
            mesh = articulate_robot_rbrt(x_rest, obs, gforce=gforce, com=com)
    else:
        mesh = x_rest.copy()
    return mesh
