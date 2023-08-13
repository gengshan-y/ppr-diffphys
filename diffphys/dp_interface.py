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

    def add_nn_modules(self):
        super().add_nn_modules()
        self.kinemtics_proxy = KinemticsProxy(self.object_field, self.scene_field)
        del self.root_pose_mlp
        del self.joint_angle_mlp
        self.root_pose_mlp = lambda x: self.kinemtics_proxy(x)
        self.joint_angle_mlp = (
            lambda x: self.kinemtics_proxy.object_field.warp.articulation.get_vals(
                x, return_so3=True
            )
        )

    def init_global_q(self):
        pass

    def get_lr_dict(self):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]

        param_lr_startwith, param_lr_with = super().get_lr_dict()
        param_lr_startwith.update(
            {
                "kinemtics_proxy": lr_base,
            }
        )
        return param_lr_startwith, param_lr_with

    def reinit_envs(self, num_envs, wdw_length, is_eval=False, overwrite=False):
        super().reinit_envs(num_envs, wdw_length, is_eval, overwrite)
        self.env.gravity[1] = -5.0

    def preset_data(self, model_dict):
        self.scene_field = model_dict["scene_field"]
        self.object_field = model_dict["object_field"]
        self.intrinsics = model_dict["intrinsics"]

        self.data_offset = self.scene_field.camera_mlp.time_embedding.frame_offset
        self.samp_int = 1.0 / 30
        self.gt_steps = self.data_offset[-1] - 1
        self.gt_steps_visible = self.gt_steps
        self.max_steps = int(self.samp_int * self.gt_steps / self.dt)
        self.skip_factor = self.max_steps // self.gt_steps

    @torch.no_grad()
    def query_kinematics_groundtruth(self, steps_fr):
        bs, n_fr = steps_fr.shape
        steps_fr = steps_fr.reshape(-1)
        batch = {}
        batch["target_q"], batch["obj2view"] = query_q(
            steps_fr, self.object_field, self.scene_field
        )
        batch["target_ja"] = query_ja(
            steps_fr, self.object_field.warp.articulation, self.env, self.robot
        )
        # TODO this is problematic due to the discrete root pose
        # batch['target_qd'] = self.compute_gradient(self.pred_est_q,  steps_fr.clone()) / self.samp_int
        # batch['target_jad']= self.compute_gradient(self.pred_est_ja, steps_fr.clone()) / self.samp_int
        batch["target_qd"] = torch.zeros_like(batch["target_q"])[..., :6]
        batch["target_jad"] = torch.zeros_like(batch["target_ja"])
        batch["ks"] = self.intrinsics.get_vals(steps_fr.reshape(-1).long())
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.reshape((bs, n_fr) + shape[1:])
        return batch

    def override_states(self):
        self.kinemtics_proxy.override_states(self.object_field, self.scene_field)

    def override_states_inv(self):
        self.kinemtics_proxy.override_states_inv(self.object_field, self.scene_field)

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
        batch = self.query_kinematics_groundtruth(steps_fr)
        target_q, target_ja, target_qd, target_jad = (
            batch["target_q"],
            batch["target_ja"],
            batch["target_qd"],
            batch["target_jad"],
        )
        self.target_q_vis = target_q[:, self.frame2step].clone()
        self.obj2view_vis = batch["obj2view"][:, self.frame2step].clone()
        self.ks_vis = batch["ks"][:, self.frame2step].clone()

        target_position, target_velocity, self.target_trajs = self.fk_pos_vel(
            target_q[:, self.frame2step],
            target_ja[:, self.frame2step],
            target_qd[:, self.frame2step],
            target_jad[:, self.frame2step],
        )

        (
            torques,
            queried,
            ref_ja,
            queried_qd,
            res_f,
        ) = self.get_net_pred(steps_fr)

        ref_ja, queried_q, queried_qd, torques, res_f = self.rearrange_pred(
            queried, ref_ja, queried_qd, torques, res_f
        )

        return target_position, ref_ja, queried_q, queried_qd, torques, res_f

    def get_foot_height(self, state_body_q):
        kp_idxs = [
            it
            for it, link in enumerate(self.robot.urdf.links)
            if link.name in self.robot.urdf.kp_links
        ]
        kp_idxs = [self.dict_unique_body_inv[it] for it in kp_idxs]
        foot_height = state_body_q[:, :, kp_idxs, 1]
        return foot_height


class KinemticsProxy(nn.Module):
    def __init__(self, object_field, scene_field):
        super(KinemticsProxy, self).__init__()
        self.object_field = copy.deepcopy(object_field)
        self.scene_field = copy.deepcopy(scene_field)

    def forward(self, x):
        out, _ = query_q(x, self.object_field, self.scene_field)
        return out

    def override_states(self, object_field, scene_field):
        # object_field and scene_field stores the states in the DR cycle
        self.object_field.load_state_dict(object_field.state_dict())
        self.scene_field.load_state_dict(scene_field.state_dict())

    def override_states_inv(self, object_field, scene_field):
        # # compute the diff between the current and the target
        # list_current = {}
        # for name, param in self.object_field.named_parameters():
        #     list_current[name] = param.data.clone()
        # for name, param in self.scene_field.named_parameters():
        #     list_current[name] = param.data.clone()

        # list_target = {}
        # for name, param in object_field.named_parameters():
        #     list_target[name] = param.data.clone()
        # for name, param in scene_field.named_parameters():
        #     list_target[name] = param.data.clone()

        # for name in list_current.keys():
        #     diff = list_current[name] - list_target[name]
        #     if diff.norm() > 0:
        #         print(name, diff.norm())

        object_field.load_state_dict(self.object_field.state_dict())
        scene_field.load_state_dict(self.scene_field.state_dict())


def query_q(steps_fr, object_field, scene_field):
    """
    bs,T
    robot2world = scale(bg2world @ bg2view^-1 @ root2view @ se3)
    """
    bs = steps_fr.shape[0]
    data_offset = scene_field.camera_mlp.time_embedding.frame_offset
    vidid, _ = fid_reindex(steps_fr, len(data_offset) - 1, data_offset)
    vidid = vidid.long()

    # query obj/scene to view
    obj_to_view = object_field.get_camera(frame_id=steps_fr.reshape(-1).long())
    scene_to_view = scene_field.get_camera(frame_id=steps_fr.reshape(-1).long())

    # urdf to object
    orient = object_field.warp.articulation.orient
    orient = dqtorch.quaternion_to_matrix(orient)
    shift = object_field.warp.articulation.shift
    urdf_to_object = torch.cat([orient, shift[..., None]], -1)
    urdf_to_object = urdf_to_object.view(1, 3, 4)
    urdf_to_object = torch.cat([urdf_to_object, obj_to_view[:1, -1:]], -2)

    # urdf to view/scene
    urdf_to_view = obj_to_view @ urdf_to_object
    urdf_to_scene = scene_to_view.inverse() @ urdf_to_view

    # scene rectification
    scene_to_world = scene_field.get_field2world(inst_id=vidid.reshape(-1))
    urdf_to_world = scene_to_world @ urdf_to_scene

    # cv to gl coords
    cv2gl = torch.eye(4, device=urdf_to_world.device)
    cv2gl[1, 1] = -1
    cv2gl[2, 2] = -1
    urdf_to_world = cv2gl[None] @ urdf_to_world

    # scale: from object to urdf
    urdf_scale = (
        object_field.logscale.exp() / object_field.warp.articulation.logscale.exp()
    )
    urdf_to_world = urdf_to_world.clone()
    urdf_to_world[..., :3, 3] *= urdf_scale

    urdf_to_view = urdf_to_view.clone()
    urdf_to_view[..., :3, 3] *= urdf_scale

    # rearrange
    urdf_to_world_vec = se3_mat2vec(urdf_to_world)  # xyzw
    urdf_to_world_vec = urdf_to_world_vec.view(bs, -1)
    urdf_to_view = urdf_to_view.clone().view(bs, 4, 4)
    return urdf_to_world_vec, urdf_to_view


def query_ja(steps_fr, nerf_body_rts, env, robot):
    """
    bs,T
    """
    bs = steps_fr.shape[0]
    data_offset = nerf_body_rts.time_embedding.frame_offset
    inst_id, _ = fid_reindex(steps_fr, len(data_offset) - 1, data_offset)
    inst_id = inst_id.long()

    # pred joint angles
    pred_joints = nerf_body_rts.get_vals(steps_fr.reshape(-1), return_so3=True)
    pred_joints = pred_joints.view(bs, -1)

    # update joint coordinates
    rel_rest_joints = nerf_body_rts.compute_rel_rest_joints(inst_id=inst_id)
    rel_rest_joints = rel_rest_joints / nerf_body_rts.logscale.exp()
    rel_rest_rmat = nerf_body_rts.local_rest_coord[None, :, :3, :3].repeat(bs, 1, 1, 1)
    rel_rest_rvec = dqtorch.matrix_to_quaternion(rel_rest_rmat)
    rel_rest_coords = torch.cat([rel_rest_joints, rel_rest_rvec[..., [1, 2, 3, 0]]], -1)

    # set first joint to identity
    zero_ones = torch.zeros_like(rel_rest_coords[:, :1])
    zero_ones[:, 0, -1] = 1
    joint_X_p = torch.cat([zero_ones, rel_rest_coords], 1)
    joint_X_p = joint_X_p.view(-1, 7)

    # B+1, 7 coordinates of end effectors
    env.joint_X_p = wp.from_torch(joint_X_p, dtype=wp.transform)
    return pred_joints
