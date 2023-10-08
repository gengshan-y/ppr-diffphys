import os, sys
import time
import torch
import torch.nn as nn
import pdb
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

from diffphys.dataloader import parse_amp
from diffphys.robot import URDFRobot
from diffphys.urdf_utils import articulate_robot_rbrt_batch, articulate_robot
from diffphys.geom_utils import (
    se3_vec2mat,
    se3_mat2vec,
    fid_reindex,
)

from warp.sim.articulation import eval_fk
from diffphys.import_urdf import parse_urdf
from diffphys.integrator_euler import SemiImplicitIntegrator
from diffphys.dp_utils import (
    rotate_frame,
    rotate_frame_vel,
    compute_com,
    reduce_loss,
    se3_loss,
    articulate_robot_rbrt,
    remove_nan,
    bullet2gl,
    compose_delta,
)
from diffphys.torch_utils import TimeMLPOld, TimeMLPWrapper, CameraMLPWrapper

sys.path.append("%s/../../../../" % os.path.dirname(__file__))
try:
    from lab4d.utils.numpy_utils import interp_wt
    from lab4d.engine.train_utils import match_param_name
except:
    from diffphys.lab4d_utils import interp_wt, match_param_name

import warp as wp

wp.init()


def get_local_rank():
    try:
        return int(os.environ["LOCAL_RANK"])
    except:
        # print("LOCAL_RANK not found, set to 0")
        return 0


class phys_model(nn.Module):
    def __init__(self, opts, dataloader, dt=5e-4, device="cuda"):
        super(phys_model, self).__init__()
        self.opts = opts
        logname = "%s-%s" % (opts["seqname"], opts["logname"])
        self.save_dir = os.path.join(opts["logroot"], logname)

        self.total_iters = (
            int(opts["num_rounds"] * opts["iters_per_round"] * opts["ratio_phys_cycle"])
            + opts["warmup_iters"]
            + 1
        )
        self.progress = 0

        self.dt = dt
        self.device = device
        self.preset_data(dataloader)

        data_dir = "%s/../" % os.path.dirname(__file__)
        if opts["urdf_template"] == "a1":
            urdf_path = "%s/data/urdf_templates/a1/urdf/a1.urdf" % data_dir
            in_bullet = True
            kp = 220.0
            kd = 2.0
            shape_ke = 1.0e4
            shape_kd = 0
        elif opts["urdf_template"] == "laikago":
            urdf_path = "%s/data/urdf_templates/laikago/laikago.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 16000.0
            self.joint_attach_kd = 200.0
            kp = 220.0
            kd = 2.0
            shape_ke = 1.0e4
            shape_kd = 0
        elif opts["urdf_template"] == "quad":
            urdf_path = "%s/data/urdf_templates/quad.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 8000.0
            self.joint_attach_kd = 200.0
            kp = 660.0
            kd = 5.0
            shape_ke = 1.0e4
            shape_kd = 0
            # shape_ke = 1000
            # shape_kd = 100
            # self.joint_attach_ke = 16000.
            # self.joint_attach_kd = 100.
            # kp=220.
            # kd=2.
        elif opts["urdf_template"] == "human":
            urdf_path = "%s/data/urdf_templates/human.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 8000.0
            self.joint_attach_kd = 200.0
            kp = 660.0
            kd = 5.0
            shape_ke = 1.0e4
            shape_kd = 0
            # shape_ke = 1000
            # shape_kd = 100
            # kp=20.
            # kd=2.
        else:
            raise NotImplementedError
        self.in_bullet = in_bullet
        self.robot = URDFRobot(urdf_path=urdf_path)

        # env
        self.articulation_builder = wp.sim.ModelBuilder()
        # TODO change path
        parse_urdf(
            urdf_path,
            self.articulation_builder,
            xform=wp.transform(
                np.array((0.0, 0.417, 0.0)),
                wp.quat_from_axis_angle((1.0, 0.0, 0.0), 0.0),
            ),
            floating=True,
            density=1000,  # collision geometry needs >0 density, but this will ignore urdf mass
            armature=0.01,  # additional inertia
            stiffness=220.0,  # ke gain
            damping=2.0,  # kd gain but was set to zero somhow
            shape_ke=shape_ke,  # collsion spring/damp/friction
            shape_kd=shape_kd,
            shape_kf=1.0e2,  # sliding friction
            shape_mu=1,  # static: use a large value to make ground sticky
            limit_ke=0,  # useful when joints violating limits
            limit_kd=0,
        )
        # limit_ke=1.e+4, # useful when joints violating limits
        # limit_kd=1.e+1)

        # for human and wolf
        if hasattr(self.robot.urdf, "kp_links"):
            # make feet heavier
            name2link_idx = [
                (link.name, it) for it, link in enumerate(self.robot.urdf.links)
            ]
            dict_unique_body = dict(enumerate(self.robot.urdf.unique_body_idx))
            self.dict_unique_body_inv = {v: k for k, v in dict_unique_body.items()}
            name2link_idx = [
                (link.name, self.dict_unique_body_inv[it])
                for it, link in enumerate(self.robot.urdf.links)
                if it in dict_unique_body.values()
            ]
            name2link_idx = dict(name2link_idx)

            # re-assign mass
            for name, idx in name2link_idx.items():
                tup = self.articulation_builder.shape_geo_scale[idx]
                # make feet longer
                if name in self.robot.urdf.kp_links:
                    self.articulation_builder.shape_geo_scale[idx] = (
                        tup[0] * 2,
                        tup[1] * 2,
                        tup[2] * 2,
                    )
                    # self.articulation_builder.body_mass[idx] *= 2
                    self.articulation_builder.body_mass[idx] *= 2**3
                    self.articulation_builder.body_inertia[idx] *= 2**5

                # normalized inertia to be only a factor of geometry ~= mx^2/6
                self.articulation_builder.body_inertia[
                    idx
                ] /= self.articulation_builder.body_mass[idx]

                # initialize link weight to a different value
                tup = self.articulation_builder.shape_geo_scale[idx]
                link_weight = 1e3 * np.prod(tup)
                print("weight of %s: %f" % (name, link_weight))
                link_weight = np.max((1.0, link_weight))  # avoid sim blow up
                link_weight = np.min((5.0, link_weight))  # avoid too heavy
                # link_weight = 1.0
                self.articulation_builder.body_mass[idx] = link_weight
        else:
            # normlize inertia
            for idx in range(len(self.articulation_builder.body_mass)):
                mass = self.articulation_builder.body_mass[idx]
                self.articulation_builder.body_inertia[idx] /= mass

        self.n_dof = len(self.articulation_builder.joint_q) - 7
        self.n_links = len(self.articulation_builder.body_q)
        self.articulation_builder.joint_target_ke = [0.0] * 6 + [kp] * (
            len(self.articulation_builder.joint_target_ke) - 6
        )
        self.articulation_builder.joint_target_kd = [0.0] * 6 + [kd] * (
            len(self.articulation_builder.joint_target_ke) - 6
        )

        # integrator
        self.integrator = SemiImplicitIntegrator()

        self.target_ke = nn.Parameter(
            torch.tensor(self.articulation_builder.joint_target_ke, dtype=torch.float32)
        )
        self.target_kd = nn.Parameter(
            torch.tensor(self.articulation_builder.joint_target_kd, dtype=torch.float32)
        )
        self.body_mass = nn.Parameter(
            torch.tensor(self.articulation_builder.body_mass, dtype=torch.float32)
        )
        self.register_buffer(
            "norm_body_inertia",
            torch.tensor(self.articulation_builder.body_inertia, dtype=torch.float32),
        )

        self.add_nn_modules()

        # torch parameters
        self.init_global_q()

        # optimizer
        self.add_optimizer(opts)

        # cache queue of length 2
        self.model_cache = [None, None]
        self.optimizer_cache = [None, None]
        self.scheduler_cache = [None, None]

        self.grad_queue = {}
        # self.param_clip_startwith = {
        #     "root_pose_mlp": 10.0,
        #     "vel_mlp": 10.0,
        # }

    def init_global_q(self):
        # necessary for fk
        self.frame2step = [0]
        self.num_envs = 1
        builder = wp.sim.ModelBuilder()
        for i in range(self.num_envs):
            builder.add_rigid_articulation(self.articulation_builder)
        self.env = builder.finalize(self.device)

        # mocap data
        self.global_q = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        steps_fr = torch.tensor([[0]])

        _, _, queried_q, queried_qd, _, _ = self.get_batch_input(steps_fr)

        queried_position, _, _ = ForwardKinematics.apply(
            queried_q[:, None], queried_qd[:, None], self.env
        )  # steps, n_env, x
        foot_height = self.get_foot_height(queried_position)[0, 0]

        self.global_q = nn.Parameter(
            torch.tensor(
                [0.0, -foot_height, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32
            )
        )

    def add_nn_modules(self):
        # self.root_pose_mlp = TimeMLPOld(tscale=1.0 / self.total_frames, out_channels=6)
        # self.joint_angle_mlp = TimeMLPOld(
        #     tscale=1.0 / self.total_frames, out_channels=self.n_dof
        # )
        # self.vel_mlp = TimeMLPOld(
        #     tscale=1.0 / self.total_frames, out_channels=6 + self.n_dof
        # )
        # self.torque_mlp = TimeMLPOld(
        #     tscale=1.0 / self.total_frames, out_channels=self.n_dof
        # )
        # self.residual_f_mlp = TimeMLPOld(
        #     tscale=1.0 / self.total_frames, out_channels=6 * self.n_links
        # )

        # msm = self.get_mocap_data(np.arange(self.total_frames))
        # rtmat = np.eye(4)[None].repeat(self.total_frames, 0)
        # rtmat[:, :3, :3] = R.from_quat(msm["orn"]).as_matrix()  # xyzw
        # rtmat[:, :3, 3] = msm["pos"]
        # rtmat = rtmat.astype(np.float32)
        # self.root_pose_mlp = CameraMLPWrapper(rtmat)
        # self.root_pose_mlp.mlp_init()

        self.root_pose_mlp = TimeMLPWrapper(
            self.total_frames,
            frame_info=self.frame_info,
            out_channels=6,
            D=8,
            skips=[4],
            time_scale=0.1,
            output_scale=0.5,
        )
        self.joint_angle_mlp = TimeMLPWrapper(
            self.total_frames, frame_info=self.frame_info, out_channels=self.n_dof
        )
        self.vel_mlp = TimeMLPWrapper(
            self.total_frames,
            frame_info=self.frame_info,
            out_channels=6 + self.n_dof,
            output_scale=5.0,
        )
        self.torque_mlp = TimeMLPWrapper(
            self.total_frames, frame_info=self.frame_info, out_channels=self.n_dof
        )
        self.residual_f_mlp = TimeMLPWrapper(
            self.total_frames, frame_info=self.frame_info, out_channels=6 * self.n_links
        )

    def set_progress(self, num_iters):
        self.progress = num_iters / self.total_iters

        # root pose prior wt: steps(0->800, 1->0), range (0,1)
        loss_name = "reg_cam_prior_wt"
        anchor_x = (0, 0.5)
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, self.progress, type=type)

    def set_loss_weight(
        self, loss_name, anchor_x, anchor_y, current_steps, type="linear"
    ):
        """Set a loss weight according to the current training step

        Args:
            loss_name (str): Name of loss weight to set
            anchor_x: Tuple of optimization steps [x0, x1]
            anchor_y: Tuple of loss values [y0, y1]
            current_steps (int): Current optimization step
            type (str): Interpolation type ("linear" or "log")
        """
        if "%s_init" % loss_name not in self.opts.keys():
            self.opts["%s_init" % loss_name] = self.opts[loss_name]
        factor = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.opts[loss_name] = self.opts["%s_init" % loss_name] * factor

    @staticmethod
    def rm_module_prefix(states, prefix="module"):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[: len(prefix)] == prefix:
                i = i[len(prefix) + 1 :]
                new_dict[i] = v
        return new_dict

    def reinit_envs(self, num_envs, frames_per_wdw, is_eval=False, overwrite=False):
        self.num_envs = num_envs
        self.frames_per_wdw = frames_per_wdw
        self.steps_idx = range(
            self.steps_per_fr_interval * (self.frames_per_wdw - 1) + 1
        )
        self.steps_idx_fr = (
            torch.cuda.LongTensor(self.steps_idx) / self.steps_per_fr_interval
        )  # frames

        self.frame2step = []
        for i in range(len(self.steps_idx)):
            if i % self.steps_per_fr_interval == 0:
                self.frame2step.append(i)

        if is_eval:
            env_name = "eval_env"
            state_name = "eval_state"
        else:
            env_name = "train_env"
            state_name = "train_state"
        if hasattr(self, env_name) and not overwrite:
            self.env = getattr(self, env_name)
            self.state_steps = getattr(self, state_name)
        else:
            # delete old env if exists
            if hasattr(self, env_name):
                delattr(self, env_name)
                delattr(self, state_name)

            builder = wp.sim.ModelBuilder()
            for i in range(self.num_envs):
                builder.add_rigid_articulation(self.articulation_builder)

            # finalize env
            self.env = builder.finalize(self.device)
            self.env.ground = True

            self.env.joint_attach_ke = self.joint_attach_ke
            self.env.joint_attach_kd = self.joint_attach_kd

            # modify it to be local
            self.state_steps = []
            for i in range(len(self.steps_idx) + 1):  # add one more step for force vis
                state = self.env.state(requires_grad=True)
                self.state_steps.append(state)

            self.env.collide(self.state_steps[0])  # ground contact, call it once
            # self.env.gravity[1] = 0.0

            setattr(self, env_name, self.env)
            setattr(self, state_name, self.state_steps)

    def preset_data(self, dataloader):
        if hasattr(dataloader, "amp_info"):
            amp_info = dataloader.amp_info

        self.frame_offset_raw = dataloader.data_info["offset"]
        self.frame_interval = dataloader.frame_interval
        self.frame_info = None

        self.total_frames = len(amp_info)
        self.steps_per_fr_interval = int(self.frame_interval / self.dt)
        print("total_frames:", self.total_frames)
        print("steps_per_fr_interval:", self.steps_per_fr_interval)

        # data query
        self.amp_info_func = scipy.interpolate.interp1d(
            np.asarray(range(self.total_frames)),
            amp_info,
            kind="linear",
            fill_value="extrapolate",
            axis=0,
        )

    def get_lr_dict(self):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["phys_learning_rate"]
        lr_explicit = lr_base * 10

        param_lr_startwith = {
            "global_q": lr_explicit,
            "target_ke": lr_explicit,
            "target_kd": lr_explicit,
            "attach_ke": lr_explicit,
            "attach_kd": lr_explicit,
            "body_mass": lr_explicit,
            "root_pose_mlp": lr_base,
            "joint_angle_mlp": lr_base,
            "vel_mlp": lr_base,
            "torque_mlp": lr_base,  # not used
            "residual_f_mlp": lr_base,  # not used
        }
        param_lr_with = {
            "root_pose_mlp.base_quat": lr_explicit,
        }
        return param_lr_startwith, param_lr_with

    def add_optimizer(self, opts):
        self.params_ref_list, params_list, lr_list = self.get_optimizable_param_list()

        self.optimizer = torch.optim.AdamW(
            params_list, lr=opts["phys_learning_rate"], weight_decay=1e-4
        )

        total_iters = max(2, self.total_iters)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            lr_list,
            total_iters,
            pct_start=2.0 / total_iters,  # warmup
            cycle_momentum=False,
            anneal_strategy="linear",
            final_div_factor=1e2,
            div_factor=25,
        )

    def get_optimizable_param_list(self):
        """
        Get the optimizable param list
        Returns:
            params_ref_list (List): List of params
            params_list (List): List of params
            lr_list (List): List of learning rates
        """
        param_lr_startwith, param_lr_with = self.get_lr_dict()
        params_ref_list = []
        params_list = []
        lr_list = []

        for name, p in self.named_parameters():
            matched_loose, lr_loose = match_param_name(name, param_lr_with, type="with")
            matched_strict, lr_strict = match_param_name(
                name, param_lr_startwith, type="startwith"
            )
            if matched_loose > 0:
                lr = lr_loose  # higher priority
            elif matched_strict > 0:
                lr = lr_strict
            else:
                lr = 0.0  # not found
                # print(name, "not found")
            if lr > 0:
                params_ref_list.append({name: p})
                params_list.append({"params": p})
                lr_list.append(lr)
                if get_local_rank() == 0:
                    print(name, p.shape, lr)
        return params_ref_list, params_list, lr_list

    def update(self):
        grad_dict = self.check_grad()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_dict

    def get_net_pred(self, steps_fr):
        """
        steps_fr: bs,T
        torques:  bs,T,dof
        delta_ja, bs,T,dof
        """
        # additional torques
        bs, nstep = steps_fr.shape
        torques = self.torque_mlp(steps_fr.reshape(-1))
        torques = torch.cat([torch.zeros_like(torques[:, :1].repeat(1, 6)), torques], 1)
        torques = torques.view(bs, nstep, -1)
        torques *= 0

        # residual force
        res_f = self.residual_f_mlp(steps_fr.reshape(-1))
        res_f = res_f.view(bs, nstep, -1, 6)
        res_f[..., :3] *= 10  # translational
        res_f = res_f.view(bs, nstep, -1)
        res_f *= 0

        # root pose
        # quat, trans = self.root_pose_mlp.get_vals(steps_fr.reshape(-1))
        # quat = quat[..., [1, 2, 3, 0]]  # wxyz => xyzw
        # delta_root = torch.cat([trans, quat], -1)
        delta_root = self.root_pose_mlp(steps_fr.reshape(-1))
        delta_root = delta_root.view(bs, nstep, -1)

        # delta joints from net
        delta_ja_ref = self.joint_angle_mlp(steps_fr.reshape(-1))
        delta_ja_ref = delta_ja_ref.view(bs, nstep, -1)

        # velocity prediction
        state_qd = self.vel_mlp(steps_fr.reshape(-1))
        state_qd = state_qd.view(bs, nstep, -1)
        return torques, delta_root, delta_ja_ref, state_qd, res_f

    @staticmethod
    def rearrange_pred(queried_q, queried_ja, queried_qd, torques, res_f):
        """
        est_q:       bs,T,7
        state_qd     bs,6
        """
        bs, nstep, _ = queried_q.shape

        # N, bs*...
        queried_q = torch.cat([queried_q, queried_ja], -1)
        queried_q = queried_q.permute(1, 0, 2).reshape(nstep, -1)
        queried_qd = queried_qd.permute(1, 0, 2).reshape(nstep, -1)
        ref_ja = torch.cat(
            [torch.zeros_like(queried_ja[..., :1].repeat(1, 1, 6)), queried_ja], -1
        )
        ref_ja = ref_ja.permute(1, 0, 2).reshape(nstep, -1)
        torques = torques.reshape(nstep, -1)
        res_f = res_f.reshape(nstep, -1, 6)
        return ref_ja, queried_q, queried_qd, torques, res_f

    def get_foot_height(self, state_body_q):
        mesh_pts, faces_single = articulate_robot_rbrt_batch(
            self.robot.urdf, state_body_q
        )
        foot_height = mesh_pts[..., 1].min(-1)[0]  # bs,T #TODO all foot
        return foot_height

    def compute_frame_start(self):
        frame_start = torch.Tensor(np.random.rand(self.num_envs)).to(self.device)
        frame_start = (
            (frame_start * (self.total_frames - self.frames_per_wdw)).round().long()
        )
        return frame_start

    def fk_pos_vel(self, target_q, target_ja, target_qd, target_jad):
        """
        In: nenv, step, dof
        """
        # combine targets
        target_q_at_frame = torch.cat([target_q, target_ja], -1)
        target_q_at_frame = target_q_at_frame.permute(1, 0, 2).contiguous()
        target_qd_at_frame = torch.cat([target_qd, target_jad], -1)
        target_qd_at_frame = target_qd_at_frame.permute(1, 0, 2).contiguous()
        # get traget body q
        target_qd_at_frame = convert_ppr_warp(target_qd_at_frame)
        target_body_q, target_body_qd, msm = ForwardKinematics.apply(
            target_q_at_frame, target_qd_at_frame, self.env
        )
        target_body_qd = convert_ppr_warp(target_body_qd)
        return target_body_q, target_body_qd, msm

    def get_mocap_data(self, steps_fr):
        amp_info = self.amp_info_func(steps_fr)
        msm = parse_amp(amp_info)
        bullet2gl(msm, self.in_bullet)
        return msm

    def get_batch_input(self, steps_fr):
        """
        get mocap data
        steps_fr: bs, T
        target_q:    bs,T,7
        target_ja:  bs,T,dof
        """
        # pos/orn/ref joints from data
        device = steps_fr.device

        msm = self.get_mocap_data(steps_fr.cpu().numpy())
        target_ja = torch.tensor(msm["jang"], dtype=torch.float32, device=device)
        target_pos = torch.tensor(msm["pos"], dtype=torch.float32, device=device)
        target_orn = torch.tensor(msm["orn"], dtype=torch.float32, device=device)
        target_jad = torch.tensor(msm["jvel"], dtype=torch.float32, device=device)
        target_vel = torch.tensor(msm["vel"], dtype=torch.float32, device=device)
        target_avel = torch.tensor(msm["avel"], dtype=torch.float32, device=device)

        target_q = torch.cat([target_pos[..., :], target_orn[..., :]], -1)  # bs, T, 7
        target_qd = torch.cat([target_vel[..., :], target_avel[..., :]], -1)  # bs, T, 6

        # transform to ground
        target_q = rotate_frame(self.global_q, target_q)
        target_qd = rotate_frame_vel(self.global_q, target_qd)

        target_position, target_velocity, self.target_trajs = self.fk_pos_vel(
            target_q[:, self.frame2step],
            target_ja[:, self.frame2step],
            target_qd[:, self.frame2step],
            target_jad[:, self.frame2step],
        )

        # get network outputs
        (
            torques,
            delta_q,
            delta_ja,
            queried_qd,
            res_f,
        ) = self.get_net_pred(steps_fr)

        # refine
        queried_q = compose_delta(target_q, delta_q)  # delta x target
        # queried_q = delta_q  # delta x target
        queried_ja = target_ja + delta_ja
        # queried_qd[..., :6] = queried_qd[..., :6] + target_qd

        ref_ja, queried_q, queried_qd, torques, res_f = self.rearrange_pred(
            queried_q, queried_ja, queried_qd, torques, res_f
        )

        return (target_position, ref_ja, queried_q, queried_qd, torques, res_f)

    def forward(self, frame_start=None):
        # capture requires cuda memory to be pre-allocated
        # wp.capture_begin()
        # self.graph = wp.capture_end()
        # this launch is not recorded in tape
        # wp.capture_launch(self.graph)

        # get time steps
        if frame_start is None:
            frame_start = self.compute_frame_start()
        else:
            frame_start = frame_start[: self.num_envs]
        steps_fr = frame_start[:, None] + self.steps_idx_fr[None]  # bs,T
        vidid, _ = fid_reindex(
            steps_fr[:, self.frame2step],
            len(self.frame_offset_raw) - 1,
            self.frame_offset_raw,
        )
        outseq_idx = (vidid[:, :1] - vidid) != 0

        # get a batch of ref pos/orn/joints
        beg = time.time()
        (
            target_position,
            ref_ja,
            queried_q,
            queried_qd,
            torques,
            res_f,
        ) = self.get_batch_input(steps_fr)

        # forward simulation
        res_fin = res_f.clone()
        q_init = queried_q[0]  # bs*7+dof
        q_init = q_init.reshape(-1)
        qd_init = queried_qd[0]
        if self.training:
            # decrease to zero when progress = 2/3
            noise_ratio = np.clip(1 - 1.5 * self.progress, 0, 1)
            q_init_noise = np.random.normal(size=q_init.shape, scale=0.01 * noise_ratio)
            q_init_noise = torch.tensor(q_init_noise, device=self.device)
            # only remove the noise on root translation
            q_init_noise = q_init_noise.view(self.num_envs, -1)
            q_init_noise[:, :3] = 0
            q_init_noise = q_init_noise.reshape(-1)
            q_init += q_init_noise

            # qd_init_noise = np.random.normal(
            #     size=qd_init.shape, scale=0.002 * noise_ratio
            # )
            # qd_init_noise = torch.tensor(qd_init_noise, device=self.device)
            # qd_init_noise = qd_init_noise.view(self.num_envs, -1)
            # qd_init_noise[:, :3] = 0
            # qd_init_noise = qd_init_noise.reshape(-1)
            # qd_init += qd_init_noise

        target_ke = self.target_ke[None].repeat(self.num_envs, 1).view(-1)
        target_kd = self.target_kd[None].repeat(self.num_envs, 1).view(-1)
        body_mass = self.body_mass[None].repeat(self.num_envs, 1).view(-1)
        body_inv_mass = 1.0 / body_mass
        body_inertia = (
            self.norm_body_inertia[None].repeat(self.num_envs, 1, 1, 1).view(-1, 3, 3)
        ) * body_mass[..., None, None]
        body_inv_inertia = body_inertia.inverse().contiguous()
        qd_init = convert_ppr_warp(qd_init)
        res_fin = convert_ppr_warp(res_fin)
        sim_position, sim_velocity = ForwardWarp.apply(
            q_init,
            qd_init,
            torques,
            res_fin,
            ref_ja,
            target_ke,
            target_kd,
            body_mass,
            body_inv_mass,
            body_inertia,
            body_inv_inertia,
            self,
        )
        sim_velocity = convert_ppr_warp(sim_velocity)

        # compute queried states
        queried_q = queried_q[self.frame2step].reshape(
            self.frames_per_wdw, self.num_envs, -1
        )
        queried_qd = queried_qd[self.frame2step].reshape(
            self.frames_per_wdw, self.num_envs, -1
        )
        # get control ref body q
        queried_qd = convert_ppr_warp(queried_qd)
        queried_position, queried_velocity, self.pid_ref = ForwardKinematics.apply(
            queried_q, queried_qd, self.env
        )
        queried_velocity = convert_ppr_warp(queried_velocity)
        foot_height = self.get_foot_height(queried_position)

        # compute targets and simulated states
        target_position = target_position.reshape(
            self.num_envs, self.frames_per_wdw, -1, 7
        )
        sim_position = sim_position.reshape(
            self.frames_per_wdw, self.num_envs, -1, 7
        ).permute(1, 0, 2, 3)
        sim_velocity = sim_velocity.reshape(
            self.frames_per_wdw, self.num_envs, -1, 6
        ).permute(1, 0, 2, 3)

        loss_dict = {}
        # targeting loss
        loss_traj = se3_loss(sim_position, target_position).mean(-1)
        loss_traj[outseq_idx] = 0
        loss_dict["traj"] = reduce_loss(loss_traj, clip=True)

        # # TODO compute 2D error
        # rtk = self.get_camera().detach()
        # sim_2d = project_bodies(sim_position, rtk)
        # target_2d = project_bodies(target_position, rtk)
        # # # img = plot_curves(sim_2d.cpu().numpy(), target_2d.cpu().numpy())
        # # img = plot_curves(sim_2d[:, :1].cpu().numpy(), target_2d[:, :1].cpu().numpy())
        # # cv2.imwrite("tmp/test.png", img[0])
        # loss_traj_2d = (sim_2d - target_2d).norm(2, -1).mean(-1)
        # loss_traj_2d[outseq_idx] = 0
        # focal_length = rtk[..., 3, 0]
        # loss_traj_2d = loss_traj_2d / focal_length
        # loss_dict["traj"] = loss_dict["traj"] + reduce_loss(loss_traj_2d)

        # queried state matching loss
        loss_pos_state = se3_loss(queried_position, sim_position.detach()).mean(-1)
        loss_pos_state[outseq_idx] = 0
        loss_dict["pos_state"] = reduce_loss(loss_pos_state)

        # proxy loss (used to regularize differentiable rendering)
        if self.opts["pos_distill_wt"] > 0.0:
            distilled_position = self.get_distilled_kinematics(steps_fr)
            loss_distill = se3_loss(distilled_position, sim_position.detach()).mean(-1)
            loss_distill[outseq_idx] = 0
            loss_dict["pos_distill"] = reduce_loss(loss_distill)

        # [angular,linear] => [linear,angular]
        loss_vel_state = se3_loss(queried_velocity, sim_velocity.detach()).mean(-1)
        loss_vel_state[outseq_idx] = 0
        loss_dict["vel_state"] = reduce_loss(loss_vel_state)

        # regularization
        loss_dict["reg_torque"] = torques.pow(2).mean()
        loss_dict["reg_res_f"] = res_f.pow(2).mean()
        loss_dict["reg_foot"] = foot_height.pow(2).mean()
        # loss_dict["reg_root"] = self.root_pose_mlp.compute_distance_to_prior()

        # modify weights
        # self.set_progress(self)

        # weight loss
        total_loss = 0
        for k, v in loss_dict.items():
            loss_dict[k] = v
            total_loss += loss_dict[k] * self.opts[k + "_wt"]

        # rename keys
        new_loss_dict = {}
        for k, v in loss_dict.items():
            new_loss_dict["loss_" + k] = v
        loss_dict = new_loss_dict

        if total_loss.isnan():
            pdb.set_trace()

        # print("loss: %.6f / fw time: %.2f s" % (total_loss.cpu(), time.time() - beg))
        loss_dict["total_loss"] = total_loss

        return loss_dict

    def backward(self, loss):
        loss.backward()

    @torch.no_grad()
    def query(self, img_size=None):
        x_sims = []
        x_msms = []
        x_control_refs = []  # target
        x_distilled_trajs = []
        com_k = []
        data = {}

        part_com = self.env.body_com.numpy()[..., None]
        part_mass = self.env.body_mass.numpy()
        body_mass = self.body_mass.cpu().numpy()
        for frame in range(len(self.sim_trajs)):
            sim_traj = self.sim_trajs[frame]
            target_traj = self.target_trajs[frame]
            pid_ref = self.pid_ref[frame]
            grf = self.grfs[frame]
            jaf = self.jafs[frame]

            # get com (simulated)
            com = compute_com(sim_traj, part_com, part_mass)
            com_k.append(compute_com(target_traj, part_com, part_mass))

            x_msm = articulate_robot_rbrt(self.robot.urdf, target_traj)
            x_control_ref = articulate_robot_rbrt(self.robot.urdf, pid_ref)
            x_sim = articulate_robot_rbrt(
                self.robot.urdf,
                sim_traj,
                gforce=grf,
                com=com,
                mass=body_mass,
            )

            x_sims.append(x_sim)
            x_msms.append(x_msm)
            x_control_refs.append(x_control_ref)

            if hasattr(self, "distilled_trajs"):
                distilled_traj = self.distilled_trajs[frame]
                x_distilled = articulate_robot_rbrt(self.robot.urdf, distilled_traj)
                x_distilled_trajs.append(x_distilled)
        x_sims = np.stack(x_sims, 0)
        x_msms = np.stack(x_msms, 0)
        x_control_refs = np.stack(x_control_refs, 0)

        data["sim_traj"] = x_sims  # simulation
        data["target_traj"] = x_msms  # reference trajectory
        data["control_ref"] = x_control_refs  # control target
        if len(x_distilled_trajs) > 0:
            data["distilled_traj"] = np.stack(x_distilled_trajs, 0)
        data["com_k"] = com_k

        if img_size is not None:
            data["camera"] = self.get_camera()[0].detach().cpu().numpy()
            data["img_size"] = img_size

        # max_w
        robot_mesh = articulate_robot(self.robot.urdf, use_collision=True)
        data["max_w"] = 3 * np.abs(robot_mesh.vertices[:, [0, 2]]).max()
        return data

    def get_camera(self):
        # get cameras: world to view = world to object + object to view
        obj2world = self.target_q_vis
        obj2world = se3_vec2mat(obj2world)
        world2view = self.world2view_vis
        world2view[..., 3, :] = self.ks_vis
        return world2view

    def save_checkpoint(self, steps_count):
        # move to the left
        self.model_cache[0] = self.model_cache[1]
        self.optimizer_cache[0] = self.optimizer_cache[1]
        self.scheduler_cache[0] = self.scheduler_cache[1]
        # enqueue
        self.model_cache[1] = deepcopy(self.state_dict())
        self.optimizer_cache[1] = deepcopy(self.optimizer.state_dict())
        self.scheduler_cache[1] = deepcopy(self.scheduler.state_dict())

        if get_local_rank() == 0:
            save_dict = self.model_cache[1]
            param_path = "%s/ckpt_phys_%04d.pth" % (self.save_dir, steps_count)
            torch.save(save_dict, param_path)

            # copy to latest
            latest_path = "%s/ckpt_phys_latest.pth" % (self.save_dir)
            os.system("cp %s %s" % (param_path, latest_path))
            return

    def load_checkpoint(self, model_path):
        states = torch.load(model_path, map_location="cpu")
        self.load_state_dict(states, strict=False)

    def check_grad(self, thresh=5.0):
        """Check if gradients are above a threshold

        Args:
            thresh (float): Gradient clipping threshold
        """
        # detect large gradients and reload model
        params_list = []
        for param_dict in self.params_ref_list:
            ((name, p),) = param_dict.items()
            if p.requires_grad and p.grad is not None:
                params_list.append(p)

        # check individual parameters
        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, thresh)
        if grad_norm > thresh:
            # clear gradients
            self.optimizer.zero_grad()
            if get_local_rank() == 0:
                print("large grad: %.2f, clear gradients" % grad_norm)
            # load cached model from two rounds ago
            if self.model_cache[0] is not None:
                if get_local_rank() == 0:
                    print("fallback to cached model")
                self.load_state_dict(self.model_cache[0])
                self.optimizer.load_state_dict(self.optimizer_cache[0])
                self.scheduler.load_state_dict(self.scheduler_cache[0])
            return {}

        # clip individual parameters
        grad_dict = {}
        queue_length = 10
        for param_dict in self.params_ref_list:
            ((name, p),) = param_dict.items()
            if p.requires_grad and p.grad is not None:
                grad = p.grad.reshape(-1).norm(2, -1)
                grad_dict["grad/" + name] = grad

                # # maintain a queue of grad norm, and clip outlier grads
                # matched_strict, clip_strict = match_param_name(
                #     name, self.param_clip_startwith, type="startwith"
                # )
                # if matched_strict:
                #     scale_threshold = clip_strict
                # else:
                #     continue
                scale_threshold = 5.0

                # check the gradient norm
                if name not in self.grad_queue:
                    self.grad_queue[name] = []
                if len(self.grad_queue[name]) > queue_length:
                    med_grad = torch.stack(self.grad_queue[name][:-1]).median()
                    grad_dict["grad_med/" + name] = med_grad
                    if grad > scale_threshold * med_grad:
                        torch.nn.utils.clip_grad_norm_(p, med_grad)
                        if get_local_rank() == 0:
                            print("large grad: %.2f, clear %s" % (grad, name))
                    else:
                        self.grad_queue[name].append(grad)
                        self.grad_queue[name].pop(0)
                else:
                    self.grad_queue[name].append(grad)

        return grad_dict

    def clear_grad(self):
        # clear gradients
        self.optimizer.zero_grad()
        # load cached model from two rounds ago
        if self.model_cache[0] is not None:
            if get_local_rank() == 0:
                print("fallback to cached model")
            self.load_state_dict(self.model_cache[0])
            self.optimizer.load_state_dict(self.optimizer_cache[0])
            self.scheduler.load_state_dict(self.scheduler_cache[0])


def convert_ppr_warp(tensor):
    # convert linear/angular convention
    # [linear,angular] <=> [angular,linear]
    # apply this to qd and force
    tensor = torch.cat([tensor[..., 3:6], tensor[..., 0:3], tensor[..., 6:]], -1)
    return tensor


class ForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rj_q, rj_qd, env):
        """
        rj_q:  T, bs, 7+B
        rj_qd: T, bs, 6+B / none
        frame2step: T
        """
        if rj_q.is_cuda:
            is_cuda = True
        else:
            is_cuda = False
            rj_q = rj_q.cuda()
            rj_qd = rj_qd.cuda()

        num_frames, bs, ndof = rj_q.shape
        # save variables
        ctx.num_frames = num_frames
        ctx.bs = bs
        ctx.ndof = ndof
        ctx.rj_q = []
        ctx.rj_qd = []
        ctx.state_steps = []
        for i in range(num_frames):
            state = env.state(requires_grad=True)
            ctx.state_steps.append(state)

        rj_q = rj_q.view(num_frames, -1)
        rj_qd = rj_qd.view(num_frames, -1)
        for it in range(num_frames):
            # q
            ctx.rj_q.append(wp.from_torch(rj_q[it]))
            # qd
            if rj_qd is None:
                rj_qd_sub = torch.tensor(np.zeros(bs * (ndof - 1)), device=rj_q.device)
            else:
                rj_qd_sub = rj_qd[it]
            rj_qd_sub = wp.from_torch(rj_qd_sub)
            ctx.rj_qd.append(rj_qd_sub)

        ctx.tape = wp.Tape()
        with ctx.tape:
            body_q = []
            body_qd = []
            body_q_numpy = []
            for it in range(num_frames):
                eval_fk(env, ctx.rj_q[it], ctx.rj_qd[it], None, ctx.state_steps[it])  #
                body_q_sub = wp.to_torch(ctx.state_steps[it].body_q)  # bs*-1,7
                body_q_sub = body_q_sub.reshape(bs, -1, 7)
                body_q.append(body_q_sub)
                body_q_numpy.append(body_q_sub.detach().cpu().numpy()[0])

                body_qd_sub = wp.to_torch(ctx.state_steps[it].body_qd)  # bs*-1,6
                body_qd_sub = body_qd_sub.reshape(bs, -1, 6)
                body_qd.append(body_qd_sub)
            body_q = torch.stack(body_q, 1)  # bs,T,dofs,7
            body_qd = torch.stack(body_qd, 1)

        if not is_cuda:
            body_q = body_q.cpu()
            body_qd = body_qd.cpu()

        return body_q, body_qd, body_q_numpy

    @staticmethod
    def backward(ctx, adj_body_qs, adj_body_qd, _):
        adj_body_qs = adj_body_qs.clone()
        adj_body_qd = adj_body_qd.clone()
        for it in range(ctx.num_frames):
            grad_body_q = adj_body_qs[:, it].reshape(-1, 7)  # bs, T, -1, 7
            ctx.state_steps[it].body_q.grad = wp.from_torch(
                grad_body_q, dtype=wp.transform
            )
            grad_body_qd = adj_body_qd[:, it].reshape(-1, 6)  # bs, T, -1, 7
            ctx.state_steps[it].body_qd.grad = wp.from_torch(
                grad_body_qd, dtype=wp.spatial_vector
            )

        # return adjoint w.r.t. inputs
        ctx.tape.backward()

        rj_q_grad = [
            wp.to_torch(ctx.tape.gradients[i]) for i in ctx.rj_q if i.requires_grad
        ]
        if len(rj_q_grad) > 0:
            rj_q_grad = torch.stack(rj_q_grad, 0).clone()  # T,bs*-1
            rj_q_grad = rj_q_grad.view(-1, ctx.bs, ctx.ndof)
            rj_q_grad[rj_q_grad.isnan()] = 0
            rj_q_grad[rj_q_grad > 1] = 1
            if rj_q_grad.isnan().sum() > 0:
                pdb.set_trace()
        else:
            rj_q_grad = None

        rj_qd_grad = [
            wp.to_torch(ctx.tape.gradients[i]) for i in ctx.rj_qd if i.requires_grad
        ]
        if len(rj_qd_grad) > 0:
            rj_qd_grad = torch.stack(rj_qd_grad, 0).clone()  # T,bs*-1
            rj_qd_grad = rj_qd_grad.view(-1, ctx.bs, ctx.ndof - 1)
            rj_qd_grad[rj_qd_grad.isnan()] = 0
            rj_qd_grad[rj_qd_grad > 1] = 1
            if rj_qd_grad.isnan().sum() > 0:
                pdb.set_trace()
        else:
            rj_qd_grad = None

        ctx.tape.zero()
        return (rj_q_grad, rj_qd_grad, None)


@wp.kernel
def wp_add(
    a: wp.array(dtype=wp.spatial_vector),
    b: wp.array(dtype=wp.spatial_vector),
):
    # get thread index
    tid = wp.tid()

    # write result back to memory
    a[tid] = a[tid] + b[tid]


class ForwardWarp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_init,
        qd_init,
        torques,
        res_f,
        refs,
        target_ke,
        target_kd,
        body_mass,
        body_inv_mass,
        body_inertia,
        body_inv_inertia,
        self,
    ):
        """
        input: torch tensors to as inputs to warp
            q_init: bs*(7+B)
            qd_init:bs*(6+B)
            torques:bs*(6+B), zeros for first 6
            res_f:  bs*(B+1),6
            refs:   bs*(6+B), zeros for first 6
        output: torch tensors that are the output of warp
            wp_pos: T,bs*(B+1),7
            wp_vel: T,bs*(B+1),7
        """
        # input: torch to wp
        ctx.q_init = wp.from_torch(q_init)
        ctx.qd_init = wp.from_torch(qd_init)
        ctx.torques = [wp.from_torch(i) for i in torques]  # very slow
        ctx.res_f = [
            wp.from_torch(i, dtype=wp.spatial_vector) for i in res_f
        ]  # very slow
        ctx.refs = [wp.from_torch(i) for i in refs]
        ctx.target_ke = wp.from_torch(target_ke)
        ctx.target_kd = wp.from_torch(target_kd)

        # mass
        ctx.body_mass = wp.from_torch(body_mass)
        ctx.body_inv_mass = wp.from_torch(body_inv_mass)
        ctx.body_inertia = wp.from_torch(body_inertia, dtype=wp.mat33)
        ctx.body_inv_inertia = wp.from_torch(body_inv_inertia, dtype=wp.mat33)

        # aux
        ctx.self = self

        # forward
        ctx.tape = wp.Tape()
        with ctx.tape:
            self.env.joint_target_kd = ctx.target_kd
            self.env.joint_target_ke = ctx.target_ke
            self.env.body_mass = ctx.body_mass
            self.env.body_inv_mass = ctx.body_inv_mass
            self.env.body_inertia = ctx.body_inertia
            self.env.body_inv_inertia = ctx.body_inv_inertia

            # assign initial states
            eval_fk(self.env, ctx.q_init, ctx.qd_init, None, self.state_steps[0])

            # simulate
            self.grfs = []
            self.jafs = []
            for step in self.steps_idx:
                self.state_steps[step].clear_forces()

                self.env.joint_target = ctx.refs[step]
                self.env.joint_act = ctx.torques[step]

                # TODO force is modified in place
                wp.launch(
                    kernel=wp_add,
                    dim=len(ctx.res_f[step]),
                    inputs=[self.state_steps[step].body_f, ctx.res_f[step]],
                    device=self.device,
                )

                grf, jaf = self.integrator.simulate(
                    self.env,
                    self.state_steps[step],
                    self.state_steps[step + 1],
                    self.dt,
                )
                # print(step)
                # print(self.state_steps[step].body_f.numpy().max())
                if step in self.frame2step:
                    # accumulate force to body
                    self.grfs.append(grf)
                    self.jafs.append(jaf)

            # get states
            self.sim_trajs = []
            wp_pos = []
            wp_vel = []
            num_coords = self.state_steps[0].body_q.shape[0] // self.num_envs
            for step in self.frame2step:
                # for vis
                obs = self.state_steps[step].body_q
                self.sim_trajs.append(obs.numpy()[:num_coords])
                wp_pos.append(wp.to_torch(obs))
                wp_vel.append(wp.to_torch(self.state_steps[step].body_qd))
            wp_pos = torch.stack(wp_pos, 0)
            wp_vel = torch.stack(wp_vel, 0)
        return wp_pos, wp_vel

    @staticmethod
    def backward(ctx, adj_body_qs, adj_body_qd):
        """
        input: gradient to
            pos_end, N,13,7
        output: gradient to
            q_init, qd_init, actions
        """
        # grad: torch to wp
        adj_body_qs = adj_body_qs.clone()
        adj_body_qd = adj_body_qd.clone()
        grad_threshold = 1.0
        self = ctx.self
        for frame, step in enumerate(self.frame2step):
            # print("save grad from frame: %d to step: %d" % (step, frame))
            self.state_steps[step].body_q.grad = wp.from_torch(
                adj_body_qs[frame], dtype=wp.transform
            )
            self.state_steps[step].body_qd.grad = wp.from_torch(
                adj_body_qd[frame], dtype=wp.spatial_vector
            )

        # return adjoint w.r.t. inputs
        # be careful, this can modify the value of input gradients
        ctx.tape.backward()

        # # zero large grads
        # grad = [wp.to_torch(v) for k, v in ctx.tape.gradients.items()]
        # grad_norm = torch.cat([i.reshape(-1) for i in grad]).norm(2, -1)
        # max_grad = torch.cat([i.reshape(-1) for i in grad]).max()
        # print("max grad in warp: %.6f" % max_grad)
        # if max_grad > grad_threshold:
        #     print("large grad in warp backward: %.6f, marked" % max_grad)
        # if grad_norm > grad_threshold or grad_norm.isnan():
        #     # if grad_norm.isnan():
        #     #     pdb.set_trace()
        #     #     print(adj_body_qs.reshape(21, 64, 26, -1)[0, :, 0, 0])

        #     print("large grad in warp backward: %.2f, clear gradients" % grad_norm)
        #     ctx.self.clear_grad()
        #     for k, v in ctx.tape.gradients.items():
        #         v.zero_()

        if ctx.q_init.requires_grad:
            q_init_grad = wp.to_torch(ctx.tape.gradients[ctx.q_init]).clone()
            remove_nan(q_init_grad, self.num_envs)
        else:
            q_init_grad = None
            print("q_init does not require grad")

        if ctx.qd_init.requires_grad:
            qd_init_grad = wp.to_torch(ctx.tape.gradients[ctx.qd_init]).clone()
            remove_nan(qd_init_grad, self.num_envs)
        else:
            qd_init_grad = None
            print("qd_init does not require grad")

        refs_grad = [
            wp.to_torch(ctx.tape.gradients[i]) for i in ctx.refs if i.requires_grad
        ]
        if len(refs_grad) > 0:
            refs_grad = torch.stack(refs_grad, 0).clone()
            remove_nan(refs_grad, self.num_envs)
        else:
            refs_grad = None
            print("refs does not require grad")

        torques_grad = [
            wp.to_torch(ctx.tape.gradients[i]) for i in ctx.torques if i.requires_grad
        ]
        if len(torques_grad) > 0:
            torques_grad = torch.stack(torques_grad, 0).clone()
            remove_nan(torques_grad, self.num_envs)
        else:
            torques_grad = None
            print("torques does not require grad")

        res_f_grad = [
            wp.to_torch(ctx.tape.gradients[i]) for i in ctx.res_f if i.requires_grad
        ]
        if len(res_f_grad) > 0:
            res_f_grad = torch.stack(res_f_grad, 0).clone()
            remove_nan(res_f_grad, self.num_envs)
        else:
            res_f_grad = None
            print("res_f does not require grad")

        if ctx.target_kd.requires_grad:
            target_kd_grad = wp.to_torch(ctx.tape.gradients[ctx.target_kd]).clone()
            remove_nan(target_kd_grad, self.num_envs)
        else:
            target_kd_grad = None
            print("target_kd does not require grad")

        if ctx.target_ke.requires_grad:
            target_ke_grad = wp.to_torch(ctx.tape.gradients[ctx.target_ke]).clone()
            remove_nan(target_ke_grad, self.num_envs)
        else:
            target_ke_grad = None
            print("target_ke does not require grad")

        if ctx.body_mass.requires_grad:
            body_mass_grad = wp.to_torch(ctx.tape.gradients[ctx.body_mass]).clone()
            remove_nan(body_mass_grad, self.num_envs)
        else:
            body_mass_grad = None
            print("body_mass does not require grad")

        if ctx.body_inv_mass.requires_grad:
            body_inv_mass_grad = wp.to_torch(
                ctx.tape.gradients[ctx.body_inv_mass]
            ).clone()
            remove_nan(body_inv_mass_grad, self.num_envs)
        else:
            body_inv_mass_grad = None
            print("body_inv_mass does not require grad")

        if ctx.body_inertia.requires_grad:
            body_inertia_grad = wp.to_torch(
                ctx.tape.gradients[ctx.body_inertia]
            ).clone()
            remove_nan(body_inertia_grad, self.num_envs)
        else:
            body_inertia_grad = None
            print("body_inertia does not require grad")

        if ctx.body_inv_inertia.requires_grad:
            body_inv_inertia_grad = wp.to_torch(
                ctx.tape.gradients[ctx.body_inv_inertia]
            ).clone()
            remove_nan(body_inv_inertia_grad, self.num_envs)
        else:
            body_inv_inertia_grad = None
            print("body_inv_inertia does not require grad")

        ctx.tape.zero()
        return (
            q_init_grad,
            qd_init_grad,
            torques_grad,
            res_f_grad,
            refs_grad,
            target_ke_grad,
            target_kd_grad,
            body_mass_grad,
            body_inv_mass_grad,
            body_inertia_grad,
            body_inv_inertia_grad,
            None,
        )
