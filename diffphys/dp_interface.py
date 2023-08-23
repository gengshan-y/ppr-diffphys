import pdb
import copy
import torch
import warp as wp
import numpy as np
import torch.nn as nn
import dqtorch

from diffphys.dp_model import phys_model, ForwardKinematics
from diffphys.geom_utils import fid_reindex, se3_mat2vec, quaternion_to_axis_angle
from diffphys.torch_utils import compute_gradient
from diffphys.dp_utils import compose_delta
from diffphys.torch_utils import TimeMLPOld


class phys_interface(phys_model):
    def __init__(self, opts, dataloader, dt=5e-4, device="cuda"):
        super(phys_interface, self).__init__(opts, dataloader, dt, device)

    def preset_data(self, model_dict):
        # not optimized except for the scale
        self.scene_field = model_dict["scene_field"]
        self.object_field = model_dict["object_field"]
        self.intrinsics = model_dict["intrinsics"]

        self.frame_offset_raw = self.scene_field.frame_offset_raw
        self.frame_interval = 0.1

        # number of observations at end points and steps in each interval
        self.total_frames = self.frame_offset_raw[-1]
        self.steps_per_fr_interval = int(self.frame_interval / self.dt)
        print("total_frames:", self.total_frames)
        print("steps_per_fr_interval:", self.steps_per_fr_interval)

    def add_nn_modules(self):
        super().add_nn_modules()
        # updated to minimize diff physics loss
        self.kinematics_proxy = KinematicsProxy(
            self.object_field,
            self.scene_field,
            self.root_pose_mlp,
            self.joint_angle_mlp,
        )
        # distilled from physics to regularize diff rendering
        self.kinematics_distilled = KinematicsProxy(
            self.object_field,
            self.scene_field,
            self.root_pose_mlp,
            self.joint_angle_mlp,
        )
        del self.root_pose_mlp
        del self.joint_angle_mlp
        self.root_pose_mlp = lambda x: self.kinematics_proxy(x)
        self.joint_angle_mlp = lambda x: self.kinematics_proxy.get_joint_angles(x)

        self.root_pose_distilled = lambda x: self.kinematics_distilled(x)
        self.joint_angle_distilled = (
            lambda x: self.kinematics_distilled.get_joint_angles(x)
        )

        # def vel_mlp(steps_fr):
        #     # def get_so3_xyz(steps_fr):
        #     #     xyz_quat = self.root_pose_mlp(steps_fr)
        #     #     xyz = xyz_quat[..., :3]
        #     #     quat = xyz_quat[..., 3:]
        #     #     so3 = quaternion_to_axis_angle(quat)
        #     #     return torch.cat([so3, xyz], -1)

        #     def get_xyz(steps_fr):
        #         xyz_quat = self.root_pose_mlp(steps_fr)
        #         xyz = xyz_quat[..., :3]
        #         return xyz

        #     # def get_joint_angles(steps_fr):
        #     #     joint_angles = self.joint_angle_mlp(steps_fr)
        #     #     joint_angles = joint_angles.view(joint_angles.shape[:-2] + (-1,))
        #     #     return joint_angles

        #     # OOM due to 75d output
        #     # NOTE: in warp state_qd is v_so3, v_xyz, v_joints, state_q is xyz, quat, joints
        #     state_qd = compute_gradient(get_xyz, steps_fr) / self.frame_interval
        #     # state_jad = compute_gradient(get_joint_angles, steps_fr) / self.frame_interval
        #     state_jad = self.jaq_mlp(steps_fr)
        #     state_qd = torch.cat([state_jad[..., :3], state_qd, state_jad[..., 3:]], -1)
        #     return state_qd

        # self.jaq_mlp = TimeMLPOld(
        #     tscale=1.0 / self.total_frames, out_channels=3 + self.n_dof
        # )
        # del self.vel_mlp
        # self.vel_mlp = vel_mlp

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
        lr_explicit = lr_base * 50
        lr_zero = 0.0

        param_lr_startwith, param_lr_with = super().get_lr_dict()
        param_lr_startwith.update(
            {
                "object_field": lr_zero,
                "scene_field": lr_zero,
                "intrinsics": lr_zero,
                "kinematics_proxy.delta_root_mlp": lr_base,
                "kinematics_proxy.delta_joint_angle_mlp": lr_base,
                "kinematics_distilled.delta_root_mlp": lr_base,
                "kinematics_distilled.delta_joint_angle_mlp": lr_base,
                # "jaq_mlp": lr_base,
            }
        )
        param_lr_with.update(
            {
                # "kinematics_proxy.object_field.logscale": lr_explicit,
                # "kinematics_proxy.object_field.camera_mlp.base_quat": lr_explicit,
                "kinematics_proxy.object_field.warp.articulation.logscale": lr_explicit,
                "kinematics_proxy.object_field.warp.articulation.shift": lr_explicit,
                "kinematics_proxy.object_field.warp.articulation.orient": lr_explicit,
                "kinematics_distilled.object_field.warp.articulation.logscale": lr_explicit,
                "kinematics_distilled.object_field.warp.articulation.shift": lr_explicit,
                "kinematics_distilled.object_field.warp.articulation.orient": lr_explicit,
                # "kinematics_proxy.scene_field.logscale": lr_explicit,
                # "kinematics_proxy.scene_field.camera_mlp.base_quat": lr_explicit,
                "object_field.logscale": lr_explicit,
                "scene_field.logscale": lr_explicit,
            }
        )
        return param_lr_startwith, param_lr_with

    # def reinit_envs(self, num_envs, frames_per_wdw, is_eval=False, overwrite=False):
    #     super().reinit_envs(num_envs, frames_per_wdw, is_eval, overwrite)
    #     self.env.gravity[1] = -5.0

    def query_kinematics_groundtruth(self, steps_fr):
        bs, n_fr = steps_fr.shape
        steps_fr = steps_fr.reshape(-1)
        batch = {}
        batch["target_q"], batch["obj2view"] = query_q(
            steps_fr, self.object_field, self.scene_field
        )
        # with torch.no_grad():
        batch["target_ja"] = query_ja(
            steps_fr, self.object_field.warp.articulation, self.env, self.robot
        )
        batch["target_qd"] = torch.zeros_like(batch["target_q"])[..., :6]
        batch["target_jad"] = torch.zeros_like(batch["target_ja"])
        batch["ks"] = self.intrinsics.get_vals(steps_fr.reshape(-1).long())
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.reshape((bs, n_fr) + shape[1:])
        return batch

    def override_states(self):
        self.kinematics_proxy.override_states(self.object_field, self.scene_field)
        self.kinematics_distilled.override_states(self.object_field, self.scene_field)

    # def override_states_inv(self):
    #     self.kinematics_proxy.override_states_inv(self.object_field, self.scene_field)

    def compute_frame_start(self):
        frame_start = torch.Tensor(np.random.rand(self.num_envs)).to(self.device)
        frame_start_all = []
        for vidid in self.opts["phys_vid"]:
            frame_start_sub = (
                frame_start
                * (
                    self.frame_offset_raw[vidid + 1]
                    - self.frame_offset_raw[vidid]
                    - self.frames_per_wdw
                )
            ).round()
            frame_start_sub = torch.clamp(frame_start_sub, 0, np.inf).long()
            frame_start_sub += self.frame_offset_raw[vidid]
            frame_start_all.append(frame_start_sub)
        frame_start = torch.cat(frame_start_all, 0)
        rand_list = np.asarray(range(frame_start.shape[0]))
        np.random.shuffle(rand_list)
        frame_start = frame_start[rand_list[: self.num_envs]]
        return frame_start

    def get_batch_input(self, steps_fr):
        # print("object scale:", self.object_field.logscale.exp())
        # print("scene scale:", self.scene_field.logscale.exp())
        # print("proxy object scale:", self.kinematics_proxy.object_field.logscale.exp())
        # print("proxy scene scale:", self.kinematics_proxy.scene_field.logscale.exp())
        batch = self.query_kinematics_groundtruth(steps_fr)
        target_position, target_velocity, self.target_trajs = self.fk_pos_vel(
            batch["target_q"][:, self.frame2step],
            batch["target_ja"][:, self.frame2step],
            batch["target_qd"][:, self.frame2step],
            batch["target_jad"][:, self.frame2step],
        )

        self.target_q_vis = batch["target_q"][:, self.frame2step].clone()
        self.obj2view_vis = batch["obj2view"][:, self.frame2step].clone()
        self.ks_vis = batch["ks"][:, self.frame2step].clone()

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

        return (target_position, ref_ja, queried_q, queried_qd, torques, res_f)

    def get_foot_height(self, state_body_q):
        kp_idxs = [
            it
            for it, link in enumerate(self.robot.urdf.links)
            if link.name in self.robot.urdf.kp_links
        ]
        kp_idxs = [self.dict_unique_body_inv[it] for it in kp_idxs]
        foot_height = state_body_q[:, :, kp_idxs, 1]
        return foot_height

    def correct_foot_position(self, increment=0.01):
        # make sure foot is above the ground by changing object scale
        self.reinit_envs(1, frames_per_wdw=self.frame_offset_raw[-1], is_eval=True)
        while True:
            self.object_field.logscale.data += increment
            self.kinematics_proxy.object_field.logscale.data += increment
            self.kinematics_distilled.object_field.logscale.data += increment
            frame_ids = torch.arange(self.total_frames, device=self.device)
            batch = self.query_kinematics_groundtruth(frame_ids[None])
            _, _, target_trajs = self.fk_pos_vel(
                batch["target_q"],
                batch["target_ja"],
                batch["target_qd"],
                batch["target_jad"],
            )
            target_trajs = np.stack(target_trajs, 0)
            foot_height = self.get_foot_height(target_trajs[None])[0]  # N,4
            print("foot height:", foot_height.min())
            if foot_height.min() > 0.0:
                break

    def get_distilled_kinematics(self, steps_fr):
        steps_fr = steps_fr[:, self.frame2step]
        bs, nstep = steps_fr.shape
        distilled_root = self.root_pose_distilled(steps_fr.reshape(-1)).view(
            bs, nstep, -1
        )
        distilled_ja = self.joint_angle_distilled(steps_fr.reshape(-1)).view(
            bs, nstep, -1
        )

        # rearrange
        distilled_q = torch.cat([distilled_root, distilled_ja], -1)
        distilled_q = distilled_q.permute(1, 0, 2).reshape(nstep, -1)

        # forward kiematics
        distilled_q = distilled_q.reshape(self.frames_per_wdw, self.num_envs, -1)
        distilled_qd = torch.zeros_like(distilled_q[..., :-1])
        distilled_position, _, self.distilled_trajs = ForwardKinematics.apply(
            distilled_q, distilled_qd, self.env
        )
        return distilled_position


class KinematicsProxy(nn.Module):
    def __init__(
        self, object_field, scene_field, delta_root_mlp, delta_joint_angle_mlp
    ):
        super(KinematicsProxy, self).__init__()
        self.object_field = copy.deepcopy(object_field)
        self.scene_field = copy.deepcopy(scene_field)
        self.delta_root_mlp = copy.deepcopy(delta_root_mlp)
        self.delta_joint_angle_mlp = copy.deepcopy(delta_joint_angle_mlp)

    def forward(self, x):
        out, _ = query_q(x, self.object_field, self.scene_field)
        delta_q = self.delta_root_mlp(x)
        out = compose_delta(out, delta_q)
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

    def get_joint_angles(self, x):
        out = self.object_field.warp.articulation.get_vals(x, return_so3=True)
        out = out + self.delta_joint_angle_mlp(x).view(out.shape)
        return out


def query_q(steps_fr, object_field, scene_field):
    """
    bs,T
    urdf_to_world = (scene_to_world @ scene_to_view^-1) @ (object_to_view @ urdf_to_object)
    urdf_to_world = (urdf_to_world_R | urdf_to_world_t * view_to_object_scale / urdf_to_object_scale)
    fixed: urdf_to_object scale
    optimized: view_to_object scale, view_to_scene scale
    """
    bs = steps_fr.shape[0]
    frame_offset_raw = scene_field.frame_offset_raw
    vidid, _ = fid_reindex(steps_fr, len(frame_offset_raw) - 1, frame_offset_raw)
    vidid = vidid.long()

    # query scales
    view_to_obj_scale = object_field.logscale.exp()
    urdf_to_obj_scale = object_field.warp.articulation.logscale.exp()

    # query obj/scene to view (object view scale, scene view scale)
    obj_to_view = object_field.get_camera(frame_id=steps_fr.reshape(-1))
    scene_to_view = scene_field.get_camera(frame_id=steps_fr.reshape(-1))

    # urdf to object (object view scale)
    orient = object_field.warp.articulation.orient
    orient = dqtorch.quaternion_to_matrix(orient)
    shift = object_field.warp.articulation.shift / view_to_obj_scale
    urdf_to_object = torch.cat([orient, shift[..., None]], -1)
    urdf_to_object = urdf_to_object.view(1, 3, 4)
    urdf_to_object = torch.cat([urdf_to_object, obj_to_view[:1, -1:]], -2)

    # urdf to view/scene (object/scene view scale)
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
    view_to_urdf_scale = view_to_obj_scale / urdf_to_obj_scale
    urdf_to_world = urdf_to_world.clone()
    urdf_to_world[..., :3, 3] *= view_to_urdf_scale

    urdf_to_view = urdf_to_view.clone()
    urdf_to_view[..., :3, 3] *= view_to_urdf_scale

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
    frame_offset_raw = nerf_body_rts.time_embedding.frame_offset_raw
    inst_id, _ = fid_reindex(steps_fr, len(frame_offset_raw) - 1, frame_offset_raw)
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
