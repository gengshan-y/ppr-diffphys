import pdb
import copy
import torch
import warp as wp
import numpy as np
import torch.nn as nn
import dqtorch

from diffphys.dp_model import phys_model
from diffphys.geom_utils import fid_reindex, se3_mat2vec


class phys_interface(phys_model):
    def __init__(self, opts, dataloader, dt=5e-4, device="cuda"):
        super(phys_interface, self).__init__(opts, dataloader, dt, device)

    def init_global_q(self):
        self.global_q = nn.Parameter(
            torch.cuda.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        )

    def add_nn_modules(self):
        super().add_nn_modules()
        self.delta_joint_est_mlp = WarpBodyMLP(self.obj_field.warp.articulation)
        self.delta_root_mlp = WarpRootMLP(
            self.obj_field, self.delta_joint_est_mlp, self.bg_field
        )

    def reinit_envs(self, num_envs, wdw_length, is_eval=False, overwrite=False):
        super().reinit_envs(num_envs, wdw_length, is_eval, overwrite)
        self.env.gravity[1] = -5.0

    def pred_est_q(self, steps_fr):
        out, obj2view = pred_est_q(steps_fr, self.obj_field, self.bg_field)
        return out, obj2view

    def pred_est_ja(self, steps_fr):
        out = pred_est_ja(
            steps_fr, self.obj_field.warp.articulation, self.env, self.robot
        )
        return out

    def preset_data(self, model_dict):
        self.bg_field = model_dict["bg_field"]
        self.obj_field = model_dict["obj_field"]
        self.intrinsics = model_dict["intrinsics"]

        self.data_offset = self.bg_field.camera_mlp.time_embedding.frame_offset
        self.samp_int = 0.1
        self.gt_steps = self.data_offset[-1] - 1
        self.gt_steps_visible = self.gt_steps
        self.max_steps = int(self.samp_int * self.gt_steps / self.dt)
        self.skip_factor = self.max_steps // self.gt_steps

    def sample_sys_state(self, steps_fr):
        bs, n_fr = steps_fr.shape
        batch = {}
        batch["target_q"], batch["obj2view"] = self.pred_est_q(steps_fr)
        batch["target_ja"] = self.pred_est_ja(steps_fr)
        # TODO this is problematic due to the discrete root pose
        # batch['target_qd'] = self.compute_gradient(self.pred_est_q,  steps_fr.clone()) / self.samp_int
        # batch['target_jad']= self.compute_gradient(self.pred_est_ja, steps_fr.clone()) / self.samp_int
        batch["target_qd"] = torch.zeros_like(batch["target_q"])[..., :6]
        batch["target_jad"] = torch.zeros_like(batch["target_ja"])
        batch["ks"] = self.intrinsics.get_vals(steps_fr.reshape(-1).long()).reshape(
            bs, n_fr, -1
        )
        return batch

    def override_states(self):
        self.delta_root_mlp.override_states(self.obj_field, self.bg_field)
        self.delta_joint_est_mlp.override_states(self.obj_field.warp.articulation)
        # self.delta_joint_ref_mlp.override_states(self.nerf_body_rts)

    def override_states_inv(self):
        self.delta_root_mlp.override_states_inv(self.obj_field, self.bg_field)
        self.delta_joint_est_mlp.override_states_inv(self.obj_field.warp.articulation)

    def compute_frame_start(self):
        frame_start = torch.Tensor(np.random.rand(self.num_envs)).to(self.device)
        frame_start_all = []
        for vidid in self.opts["phys_vid"]:
            frame_start_sub = (
                frame_start
                * (
                    self.data_offset[vidid + 1]
                    - self.data_offset[vidid]
                    - self.wdw_length
                )
            ).round()
            frame_start_sub = torch.clamp(frame_start_sub, 0, np.inf).long()
            frame_start_sub += self.data_offset[vidid]
            frame_start_all.append(frame_start_sub)
        frame_start = torch.cat(frame_start_all, 0)
        rand_list = np.asarray(range(frame_start.shape[0]))
        np.random.shuffle(rand_list)
        frame_start = frame_start[rand_list[: self.num_envs]]
        return frame_start

    def get_batch_input(self, steps_fr):
        with torch.no_grad():
            batch = self.sample_sys_state(steps_fr)
            target_q, target_ja, target_qd, target_jad = (
                batch["target_q"],
                batch["target_ja"],
                batch["target_qd"],
                batch["target_jad"],
            )
            self.target_q_vis = target_q[:, self.frame2step].clone()
            self.obj2view_vis = batch["obj2view"][:, self.frame2step].clone()
            self.ks_vis = batch["ks"][:, self.frame2step].clone()

        target_body_q, target_body_qd, msm = self.combine_targets(
            target_q, target_ja, target_qd, target_jad
        )

        (
            torques,
            est_q,
            ref_ja,
            est_ja,
            state_qd,
            res_f,
        ) = self.get_net_pred(steps_fr)

        return (
            target_body_q,
            target_body_qd,
            msm,
            ref_ja,
            est_q,
            est_ja,
            state_qd,
            torques,
            res_f,
        )

    def get_foot_height(self, state_body_q):
        kp_idxs = [
            it
            for it, link in enumerate(self.robot.urdf.links)
            if link.name in self.robot.urdf.kp_links
        ]
        kp_idxs = [self.dict_unique_body_inv[it] for it in kp_idxs]
        foot_height = state_body_q[:, :, kp_idxs, 1]
        return foot_height


class WarpRootMLP(nn.Module):
    def __init__(self, banmo_root_mlp, banmo_body_mlp, banmo_bg_mlp):
        super(WarpRootMLP, self).__init__()
        self.root_mlp = copy.deepcopy(banmo_root_mlp)
        # self.body_mlp = copy.deepcopy(banmo_body_mlp)
        self.body_mlp = banmo_body_mlp.mlp
        self.bg_mlp = copy.deepcopy(banmo_bg_mlp)
        # self.bg_mlp.ft_bgcam=True # asssuming it's accurate enough

    def forward(self, x):
        out, _ = pred_est_q(x, self.root_mlp, self.bg_mlp)
        return out

    def override_states(self, banmo_root_mlp, banmo_bg_mlp):
        self.root_mlp.load_state_dict(banmo_root_mlp.state_dict())
        # self.body_mlp.load_state_dict(banmo_body_mlp.state_dict())
        self.bg_mlp.load_state_dict(banmo_bg_mlp.state_dict())

    def override_states_inv(self, banmo_root_mlp, banmo_bg_mlp):
        banmo_root_mlp.load_state_dict(self.root_mlp.state_dict())
        # banmo_body_mlp.load_state_dict(self.body_mlp.state_dict())
        banmo_bg_mlp.load_state_dict(self.bg_mlp.state_dict())


class WarpBodyMLP(nn.Module):
    def __init__(self, banmo_mlp):
        super(WarpBodyMLP, self).__init__()
        self.mlp = copy.deepcopy(banmo_mlp)

    def forward(self, x):
        out = self.mlp.get_vals(x.long(), return_so3=True)
        return out

    def override_states(self, banmo_mlp):
        self.mlp.load_state_dict(banmo_mlp.state_dict())

    def override_states_inv(self, banmo_mlp):
        banmo_mlp.load_state_dict(self.mlp.state_dict())


def pred_est_q(steps_fr, obj_field, bg_field):
    """
    bs,T
    robot2world = scale(bg2world @ bg2view^-1 @ root2view @ se3)
    """
    bs, n_fr = steps_fr.shape
    data_offset = bg_field.camera_mlp.time_embedding.frame_offset
    vidid, _ = fid_reindex(steps_fr, len(data_offset) - 1, data_offset)
    vidid = vidid.long()

    obj_to_view = obj_field.get_camera(frame_id=steps_fr.reshape(-1).long())
    scene_to_view = bg_field.get_camera(frame_id=steps_fr.reshape(-1).long())  # -1,3,4
    obj_to_scene = scene_to_view.inverse() @ obj_to_view

    # scene rectification
    scene_to_world = bg_field.get_field2world(inst_id=vidid.reshape(-1))
    obj_to_world = scene_to_world @ obj_to_scene

    # cv to gl coords
    cv2gl = torch.eye(4).to(obj_to_world.device)
    cv2gl[1, 1] = -1
    cv2gl[2, 2] = -1
    obj_to_world = cv2gl[None] @ obj_to_world

    obj_to_world_vec = se3_mat2vec(obj_to_world)  # xyzw
    obj_to_world_vec = obj_to_world_vec.view(bs, n_fr, -1)

    obj_to_view = obj_to_view.clone().view(bs, n_fr, 4, 4)
    return obj_to_world_vec, obj_to_view


def pred_est_ja(steps_fr, nerf_body_rts, env, robot):
    """
    bs,T
    """
    bs, n_fr = steps_fr.shape
    device = steps_fr.device
    data_offset = nerf_body_rts.time_embedding.frame_offset
    inst_id, _ = fid_reindex(steps_fr, len(data_offset) - 1, data_offset)
    inst_id = inst_id[:, 0].long()

    # pred joint angles
    pred_joints = nerf_body_rts.get_vals(steps_fr.reshape(-1).long(), return_so3=True)
    pred_joints = pred_joints.view(bs, n_fr, -1)

    # update joint locations
    rel_rest_joints = nerf_body_rts.compute_rel_rest_joints(inst_id=inst_id)
    rel_rest_joints *= 20  # TODO fix it
    zero_ones = torch.zeros((bs, nerf_body_rts.num_se3, 4), device=device)
    zero_ones[..., -1] = 1  # xyzw
    rel_rest_joints = torch.cat([rel_rest_joints, zero_ones], -1)

    # set first joitn to identity
    zero_ones = torch.zeros_like(rel_rest_joints[:, :1])
    zero_ones[:, 0, -1] = 1
    joint_X_p = torch.cat([zero_ones, rel_rest_joints], 1)
    joint_X_p = joint_X_p.view(-1, 7)

    # B+1, 7 coordinates of end effectors
    env.joint_X_p = wp.from_torch(joint_X_p, dtype=wp.transform)
    return pred_joints
