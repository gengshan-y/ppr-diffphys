import os, sys
import time
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import pdb
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R

from diffphys.dataloader import parse_amp
from diffphys.torch_utils import NeRF
from diffphys.robot import URDFRobot
from diffphys.urdf_utils import (
    articulate_robot_rbrt_batch
)
from diffphys.geom_utils import (
    se3_vec2mat,
    se3_mat2vec,
    fid_reindex,
)

from warp.sim.articulation import eval_fk
from diffphys.import_urdf import parse_urdf
from diffphys.integrator_euler import SemiImplicitIntegrator
from diffphys.dp_utils import rotate_frame, rotate_frame_vel, compute_com, clip_loss, se3_loss, can2gym2gl, remove_nan, bullet2gl

import warp as wp
wp.init()


class phys_model(nn.Module):
    def __init__(self, opts, dataloader, dt=5e-4, device="cuda"):
        super(phys_model, self).__init__()
        self.opts = opts
        logname = "%s-%s" % (opts["seqname"], opts["logname"])
        self.save_dir = os.path.join(opts["logroot"], logname)

        self.total_iters = int(
            opts["num_rounds"] * opts["iters_per_round"] * opts["ratio_phys_cycle"]
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
        elif opts["urdf_template"] == "wolf":
            urdf_path = "%s/data/urdf_templates/wolf.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 32000.0
            self.joint_attach_kd = 100.0
            kp = 2200.0
            kd = 2.0
            shape_ke = 1000
            shape_kd = 100
        elif opts["urdf_template"] == "wolf_mod":
            urdf_path = "%s/data/urdf_templates/wolf_mod.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 8000.0
            self.joint_attach_kd = 200.0
            kp = 660.0
            kd = 5.0
            shape_ke = 1000
            shape_kd = 100
            # self.joint_attach_ke = 16000.
            # self.joint_attach_kd = 100.
            # kp=220.
            # kd=2.
        elif opts["urdf_template"] == "laikago":
            urdf_path = "%s/data/urdf_templates/laikago/laikago.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 16000.0
            self.joint_attach_kd = 200.0
            kp = 220.0
            kd = 2.0
            shape_ke = 1.0e4
            shape_kd = 0
        elif opts["urdf_template"] == "human":
            urdf_path = "%s/data/urdf_templates/human.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 64000.0
            self.joint_attach_kd = 150.0  # tune this such that it would not blow up
            kp = 20.0
            kd = 2.0
            shape_ke = 1000
            shape_kd = 100
        elif opts["urdf_template"] == "human_mod":
            urdf_path = "%s/data/urdf_templates/human_mod.urdf" % data_dir
            in_bullet = False
            self.joint_attach_ke = 8000.0
            self.joint_attach_kd = 200.0
            kp = 660.0
            kd = 5.0
            shape_ke = 1000
            shape_kd = 100
            # kp=20.
            # kd=2.
        elif opts["urdf_template"] == "human_amp":
            urdf_path = "%s/data/urdf_templates/human_amp.urdf" % data_dir
            in_bullet = False
            kp = 20.0
            kd = 2.0
            shape_ke = 1000
            shape_kd = 100
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
            shape_kf=1.0e2,
            shape_mu=1,  # use a large value to make ground sticky
            limit_ke=0,  # useful when joints violating limits
            limit_kd=0,
        )
        # limit_ke=1.e+4, # useful when joints violating limits
        # limit_kd=1.e+1)

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
        # lighter trunk
        # for kp_name in ['link_136_Bauch_Y', 'link_137_Bauch.001_Y',
        #                'link_138_Brust_Y', 'link_139_Hals_Y']:
        #    kp_idx = name2link_idx[kp_name]
        #    self.articulation_builder.body_mass[kp_idx] *= 1./10

        if hasattr(self.robot.urdf, "kp_links"):
            # for human and wolf
            for i in range(len(self.articulation_builder.body_mass)):
                self.articulation_builder.body_mass[i] = 2
            for kp_name in self.robot.urdf.kp_links:
                kp_idx = name2link_idx[kp_name]
                self.articulation_builder.body_mass[kp_idx] *= 2
                tup = self.articulation_builder.shape_geo_scale[kp_idx]
                self.articulation_builder.shape_geo_scale[kp_idx] = (
                    tup[0] * 2,
                    tup[1] * 2,
                    tup[2] * 2,
                )

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

        # torch parameters
        self.init_global_q()

        self.target_ke = nn.Parameter(
            torch.cuda.FloatTensor(self.articulation_builder.joint_target_ke)
        )
        self.target_kd = nn.Parameter(
            torch.cuda.FloatTensor(self.articulation_builder.joint_target_kd)
        )
        self.body_mass = nn.Parameter(
            torch.cuda.FloatTensor(self.articulation_builder.body_mass)
        )

        self.add_nn_modules()

        # optimizer
        self.add_optimizer(opts)
        self.total_loss_hist = []

        # other hypoter parameters
        self.th_multip = 0  # seems to cause problems


    def init_global_q(self):
        self.global_q = nn.Parameter(
            torch.cuda.FloatTensor([0.0, -0.03, 0.0, 0.0, 0.0, 0.0, 1.0])
        )

    def add_nn_modules(self):
        self.torque_mlp = NeRF(
            tscale=1.0 / self.gt_steps,
            N_freqs=6,
            D=8,
            W=256,
            out_channels=self.n_dof,
            in_channels_xyz=13,
        )

        self.residual_f_mlp = NeRF(
            tscale=1.0 / self.gt_steps,
            N_freqs=6,
            D=8,
            W=256,
            out_channels=6 * self.n_links,
            in_channels_xyz=13,
        )

        self.delta_root_mlp = NeRF(
            tscale=1.0 / self.gt_steps,
            N_freqs=6,
            D=8,
            W=256,
            out_channels=6,
            in_channels_xyz=13,
        )

        self.vel_mlp = NeRF(
            tscale=1.0 / self.gt_steps,
            N_freqs=6,
            D=8,
            W=256,
            out_channels=6 + self.n_dof,
            in_channels_xyz=13,
        )

        self.delta_joint_ref_mlp = NeRF(
            tscale=1.0 / self.gt_steps,
            N_freqs=6,
            D=8,
            W=256,
            out_channels=self.n_dof,
            in_channels_xyz=13,
        )

        self.delta_joint_est_mlp = NeRF(
            tscale=1.0 / self.gt_steps,
            N_freqs=6,
            D=8,
            W=256,
            out_channels=self.n_dof,
            in_channels_xyz=13,
        )

    def set_progress(self, num_iters):
        self.progress = num_iters / self.total_iters

    @staticmethod
    def rm_module_prefix(states, prefix="module"):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[: len(prefix)] == prefix:
                i = i[len(prefix) + 1 :]
                new_dict[i] = v
        return new_dict

    def reinit_envs(self, num_envs, wdw_length, is_eval=False, overwrite=False):
        self.num_envs = num_envs
        self.wdw_length = wdw_length  # frames
        self.local_steps = range(self.skip_factor * self.wdw_length)
        self.local_steps_fr = (
            torch.cuda.LongTensor(self.local_steps) / self.skip_factor
        )  # frames
        self.wdw_length_full = len(self.local_steps)

        self.frame2step = [0]
        for i in range(len(self.local_steps) + 1):
            if (i + 1) % self.skip_factor == 0:
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
            for i in range(len(self.local_steps) + 1):
                state = self.env.state(requires_grad=True)
                self.state_steps.append(state)

            self.env.collide(self.state_steps[0])  # ground contact, call it once

            setattr(self, env_name, self.env)
            setattr(self, state_name, self.state_steps)

    def preset_data(self, dataloader):
        if hasattr(dataloader, "amp_info"):
            amp_info = dataloader.amp_info

        self.data_offset = dataloader.data_info["offset"]
        self.samp_int = dataloader.samp_int

        self.gt_steps = len(amp_info) - 1  # for k frames, need to step k-1 times
        self.gt_steps_visible = self.gt_steps
        self.max_steps = int(self.samp_int * self.gt_steps / self.dt)
        self.skip_factor = self.max_steps // self.gt_steps

        # data query
        self.amp_info_func = scipy.interpolate.interp1d(
            np.asarray(range(self.gt_steps + 1)),
            amp_info,
            kind="linear",
            fill_value="extrapolate",
            axis=0,
        )

    def add_params_to_dict(self, clip_grad=False):
        opts = self.opts
        params_list = []
        lr_dict = {}
        grad_dict = {}
        is_invalid_grad = False
        for name, p in self.named_parameters():
            params = {"params": [p]}
            params_list.append(params)
            if "bg2fg_scale" in name:
                # print('bg2fg_scale')
                # print(p)
                lr = 100 * opts["learning_rate"]  # explicit variables
                g_th = 0.1
            elif "bg2world" in name:
                lr = 0 * opts["learning_rate"]  # do not update
                g_th = 0
            elif "global_q" in name:
                lr = opts["learning_rate"]
                g_th = 100
            elif "target_ke" == name or "target_kd" == name:
                lr = 20 * opts["learning_rate"]
                g_th = 0.1
            elif "attach_ke" == name or "attach_kd" == name:
                lr = 20 * opts["learning_rate"]
                g_th = 0.1
            elif "body_mass" == name:
                lr = 20 * opts["learning_rate"]
                g_th = 0.1
            elif "torque_mlp" in name:
                lr = opts["learning_rate"]
                g_th = 1
            elif "residual_f_mlp" in name:
                lr = opts["learning_rate"]
                g_th = 1
            elif "delta_root_mlp" in name:
                lr = opts["learning_rate"]
                g_th = 1
            elif "vel_mlp" in name:
                lr = opts["learning_rate"]
                g_th = 1
            elif "delta_joint_est_mlp" in name:
                lr = opts["learning_rate"]
                g_th = 1
            elif "delta_joint_ref_mlp" in name:
                lr = opts["learning_rate"]
                g_th = 1
            else:
                lr = opts["learning_rate"]
                g_th = 1
            lr_dict[name] = lr
            if clip_grad:
                try:
                    pgrad_nan = p.grad.isnan()
                    if pgrad_nan.sum() > 0:
                        print("%s grad invalid" % name)
                        is_invalid_grad = True
                except:
                    pass
                grad_dict["grad_" + name] = clip_grad_norm_(p, g_th)

        return params_list, lr_dict, grad_dict, is_invalid_grad

    def clip_grad(self):
        """
        gradient clipping
        """
        _, _, grad_dict, is_invalid_grad = self.add_params_to_dict(clip_grad=True)

        if is_invalid_grad:
            zero_grad_list(self.parameters())

        return grad_dict

    def add_optimizer(self, opts):
        params_list, lr_dict, _, _ = self.add_params_to_dict()
        for name, lr in lr_dict.items():
            print("optimized params: %.5f/%s" % (lr, name))

        self.optimizer = torch.optim.AdamW(params_list, lr=opts["learning_rate"])
        # self.optimizer = torch.optim.SGD(
        # params_list,
        # lr=1)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            list(lr_dict.values()),
            self.total_iters,
            pct_start=0.02,  # use 2%
            cycle_momentum=False,
            anneal_strategy="linear",
            final_div_factor=1.0 / 5,
            div_factor=25,
        )
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
        # list(lr_dict.values()),
        # int(1e6), # 1000k steps
        # pct_start=0.02, # use 2%
        # cycle_momentum=False,
        # anneal_strategy='linear',
        # final_div_factor=1., div_factor = 1.,
        # )

    def update(self):
        grad_dict = self.clip_grad()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_dict

    @staticmethod
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

    @staticmethod
    def compute_gradient(fn, x):
        """
        gradient of mlp params wrt pts
        """
        x.requires_grad_(True)
        y = fn(x)

        # get gradient for each size-1 output
        gradients = []
        for i in range(y.shape[-1]):
            y_sub = y[..., i : i + 1]
            d_output = torch.ones_like(y_sub, requires_grad=False, device=y.device)
            gradient = torch.autograd.grad(
                outputs=y_sub,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients.append(gradient[..., None])
        gradients = torch.cat(gradients, -1)  # ...,input-dim, output-dim
        return gradients

    def get_net_pred(self, steps_fr):
        """
        steps_fr: bs,T
        torques:  bs,T,dof
        delta_ja, bs,T,dof
        """
        # additional torques from net
        bs, nstep = steps_fr.shape
        torques = self.torque_mlp(steps_fr.reshape(-1, 1))
        torques = torch.cat([torch.zeros_like(torques[:, :1].repeat(1, 6)), torques], 1)
        torques = torques.view(bs, nstep, -1)
        # torques *= 0

        # residual force
        res_f = self.residual_f_mlp(steps_fr.reshape(-1, 1))
        res_f = res_f.view(bs, nstep, -1, 6)
        # res_f[:,:,1:] = 0
        res_f[..., 3:6] *= 10
        # res_f *= 0
        # res_f = res_f.view(bs,nstep,-1,6)
        # res_f[:,:,:,4] = 1
        res_f = res_f.view(bs, nstep, -1)

        # delta root transoforms: G = Gd G
        delta_root = self.delta_root_mlp(steps_fr.reshape(-1, 1))
        delta_root = delta_root.view(bs, nstep, -1)

        ## ref joints from net
        # delta_ja_ref = self.delta_joint_ref_mlp(steps_fr.reshape(-1,1))
        # delta_ja_ref = delta_ja_ref.view(bs,nstep,-1)

        # delta joints from net
        delta_ja_est = self.delta_joint_est_mlp(steps_fr.reshape(-1, 1))
        delta_ja_est = delta_ja_est.view(bs, nstep, -1)
        delta_ja_ref = delta_ja_est

        vel_pred = self.vel_mlp(steps_fr.reshape(-1, 1))
        vel_pred = vel_pred.view(bs, nstep, -1)

        return torques, delta_root, delta_ja_ref, delta_ja_est, vel_pred, res_f

    @staticmethod
    def rearrange_pred(est_q, est_ja, ref_ja, state_qd, torques, res_f):
        """
        est_q:       bs,T,7
        state_qd     bs,6
        """
        bs, nstep, _ = est_q.shape

        # initial states
        state_q = torch.cat([est_q, est_ja], -1)

        # N, bs*...
        ref = torch.cat([torch.zeros_like(ref_ja[..., :1].repeat(1, 1, 6)), ref_ja], -1)
        ref = ref.permute(1, 0, 2).reshape(nstep, -1)
        state_q = state_q.permute(1, 0, 2).reshape(nstep, -1)
        state_qd = state_qd.permute(1, 0, 2).reshape(nstep, -1)
        torques = torques.reshape(nstep, -1)
        res_f = res_f.reshape(nstep, -1, 6)

        return ref, state_q, state_qd, torques, res_f

    def get_foot_height(self, state_body_q):
        mesh_pts, faces_single = articulate_robot_rbrt_batch(
            self.robot.urdf, state_body_q
        )
        foot_height = mesh_pts[..., 1].min(-1)[0]  # bs,T #TODO all foot
        return foot_height

    def compute_frame_start(self):
        frame_start = torch.Tensor(np.random.rand(self.num_envs)).to(self.device)
        frame_start = (
            (frame_start * (self.gt_steps_visible - self.wdw_length))
            .round()
            .long()
        )
        return frame_start

    def combine_targets(self, target_q, target_ja, target_qd, target_jad):
        # combine targets
        q_at_frame = torch.cat([target_q, target_ja], -1)[:, self.frame2step]
        qd_at_frame = torch.cat([target_qd, target_jad], -1)[:, self.frame2step]
        q_at_frame = q_at_frame.permute(1, 0, 2).contiguous()
        qd_at_frame = qd_at_frame.permute(1, 0, 2).contiguous()
        target_body_q, target_body_qd, msm = ForwardKinematics.apply(
            q_at_frame, qd_at_frame, self
        )
        return target_body_q, target_body_qd, msm

    def get_batch_input(self, steps_fr):
        """
        get mocap data
        steps_fr: bs, T
        target_q:    bs,T,7
        target_ja:  bs,T,dof
        """
        # pos/orn/ref joints from data
        amp_info = self.amp_info_func(steps_fr.cpu().numpy())
        msm = parse_amp(amp_info)
        bullet2gl(msm, self.in_bullet)
        target_ja = torch.cuda.FloatTensor(msm["jang"])
        target_pos = torch.cuda.FloatTensor(msm["pos"])
        target_orn = torch.cuda.FloatTensor(msm["orn"])
        target_jad = torch.cuda.FloatTensor(msm["jvel"])
        target_vel = torch.cuda.FloatTensor(msm["vel"])
        target_avel = torch.cuda.FloatTensor(msm["avel"])

        target_q = torch.cat([target_pos[..., :], target_orn[..., :]], -1)  # bs, T, 7
        target_qd = torch.cat([target_vel[..., :], target_avel[..., :]], -1)  # bs, T, 6

        # transform to ground
        target_q = rotate_frame(self.global_q, target_q)
        target_qd = rotate_frame_vel(self.global_q, target_qd)

        target_body_q, target_body_qd, msm = self.combine_targets(target_q, target_ja, target_qd, target_jad)
        
        # combine preds
        (
            torques,
            delta_q,
            delta_ja_ref,
            delta_ja_est,
            state_qd,
            res_f,
        ) = self.get_net_pred(steps_fr)

        # refine
        est_q = self.compose_delta(target_q, delta_q)  # delta x target
        est_ja = target_ja + delta_ja_est
        ref_ja = target_ja + delta_ja_ref

        return target_body_q, target_body_qd, msm, ref_ja, est_q, est_ja, state_qd, torques, res_f

    def forward(self, frame_start=None):
        # capture requires cuda memory to be pre-allocated
        # wp.capture_begin()
        # self.graph = wp.capture_end()
        # this launch is not recorded in tape
        # wp.capture_launch(self.graph)

        # get a batch of ref pos/orn/joints
        if frame_start is None:
            # get a batch of clips
            frame_start = self.compute_frame_start()
        else:
            frame_start = frame_start[: self.num_envs]

        steps_fr = frame_start[:, None] + self.local_steps_fr[None]  # bs,T
        # TODO steps that are in the same seq
        vidid, _ = fid_reindex(
            steps_fr[:, self.frame2step], len(self.data_offset) - 1, self.data_offset
        )
        outseq_idx = (vidid[:, :1] - vidid) != 0


        # compute target pos/vel
        beg = time.time()
        target_body_q, target_body_qd, self.msm, ref_ja, est_q, est_ja, state_qd, torques, res_f = \
            self.get_batch_input(steps_fr)

        ref, state_q, state_qd, torques, res_f = self.rearrange_pred(
            est_q, est_ja, ref_ja, state_qd, torques, res_f
        )

        # forward simulation
        res_fin = res_f.clone()
        q_init = state_q[0]
        qd_init = state_qd[0]
        if self.training:
            # TODO add some noise
            noise_ratio = np.clip(1 - 1.5 * self.progress, 0, 1)
            q_init_noise = np.random.normal(size=q_init.shape, scale=0.0 * noise_ratio)
            # q_init_noise = np.random.normal(size=q_init.shape,scale=0.05*noise_ratio)
            # q_init_noise = np.random.normal(size=q_init.shape,scale=0.1*noise_ratio)
            qd_init_noise = np.random.normal(
                size=qd_init.shape, scale=0.01 * noise_ratio
            )
            q_init_noise = torch.Tensor(q_init_noise).to(self.device)
            qd_init_noise = torch.Tensor(qd_init_noise).to(self.device)
            # only keep the noise on root pose
            q_init_noise = q_init_noise.view(self.num_envs, -1)
            q_init_noise[:, :3] = 0
            q_init_noise[:, 7:] = 0
            q_init_noise = q_init_noise.reshape(-1)

            q_init += q_init_noise
            # qd_init += qd_init_noise

        target_ke = self.target_ke[None].repeat(self.num_envs, 1).view(-1)
        target_kd = self.target_kd[None].repeat(self.num_envs, 1).view(-1)
        body_mass = self.body_mass[None].repeat(self.num_envs, 1).view(-1)
        body_qs, body_qd = ForwardWarp.apply(
            q_init,
            qd_init,
            torques,
            res_fin,
            ref,
            target_ke,
            target_kd,
            body_mass,
            self,
        )

        ## compute state pos/vel: bs, T, K,7/6
        state_q = state_q[self.frame2step].reshape(
            self.wdw_length + 1, self.num_envs, -1
        )
        state_qd = state_qd[self.frame2step].reshape(
            self.wdw_length + 1, self.num_envs, -1
        )
        state_body_q, state_body_qd, self.tstate = ForwardKinematics.apply(
            state_q, state_qd, self
        )

        # make sure the feet is above the ground
        foot_height = self.get_foot_height(state_body_q)

        total_loss = 0
        # root loss
        # root_target = torch.cuda.FloatTensor([1,0,0],device=self.device)
        # root_traj = body_qs[-1,0,:3]

        # root_target = torch.cuda.FloatTensor([[0,0.45,0]],device=self.device)
        # root_traj = body_qs[:, 0, :3] # S, 13, 7

        body_target = target_body_q.reshape(self.num_envs, self.wdw_length + 1, -1, 7)
        body_traj = body_qs.reshape(self.wdw_length + 1, self.num_envs, -1, 7).permute(
            1, 0, 2, 3
        )
        body_vel = body_qd.reshape(self.wdw_length + 1, self.num_envs, -1, 6).permute(
            1, 0, 2, 3
        )

        loss_root = se3_loss(body_traj, body_target).mean(-1)
        loss_root[outseq_idx] = 0
        loss_root = clip_loss(loss_root, 0.02 * self.th_multip)
        total_loss += 0.1 * loss_root

        ## body loss
        # body_traj = body_qs[:, 1:] # S, 13, 7
        # body_target = body_qs[:1,1:].detach() # first frame
        # loss_pose = (body_traj - body_target).pow(2).sum()
        # total_loss += loss_pose

        ## velocity loss
        loss_vel = se3_loss(state_body_qd, target_body_qd, rot_ratio=0).mean(-1)[:, 1:]
        loss_vel[outseq_idx[:, 1:]] = 0
        loss_vel = clip_loss(loss_vel, 20 * self.th_multip)
        # total_loss += loss_vel*1e-5

        ## vel input loss
        # loss_vel = (vel_pred[:,:1,:6] - pred_qd).norm(2,1).mean()
        # loss_vel +=(vel_pred[:,:1,6:] - pred_joint_qd).norm(2,1).mean()
        # loss_vel *= 1e-4
        # total_loss += loss_vel

        # state matching
        loss_root_state = se3_loss(state_body_q, body_traj).mean(-1)[:, 1:]
        loss_root_state[outseq_idx[:, 1:]] = 0
        loss_root_state = clip_loss(loss_root_state, 0.02 * self.th_multip)
        total_loss += 1e-1 * loss_root_state

        loss_vel_state = se3_loss(state_body_qd, body_vel).mean(-1)[:, 1:]
        loss_vel_state[outseq_idx[:, 1:]] = 0
        loss_vel_state = clip_loss(loss_vel_state, 20 * self.th_multip)
        total_loss += loss_vel_state * 1e-5

        ## reg
        torque_reg = torques.pow(2).mean()
        total_loss += torque_reg * 1e-5

        res_f_reg = res_f.pow(2).mean()
        # total_loss += res_f_reg*1e-2
        total_loss += res_f_reg * 5e-5
        # total_loss += res_f_reg*1e-5

        # delta_joint_ref_reg = delta_ja_ref.pow(2).mean()
        # total_loss += delta_joint_ref_reg*1e-4
        
        foot_reg = foot_height.pow(2).mean()
        # total_loss += foot_reg * 1e-4

        if total_loss.isnan():
            pdb.set_trace()

        # loss
        print("loss: %.4f / fw time: %.2f s" % (total_loss.cpu(), time.time() - beg))
        if len(self.total_loss_hist) > 0:
            his_med = torch.stack(self.total_loss_hist, 0).median()
            print(his_med)
            if total_loss > his_med * 10:
                total_loss.zero_()
            else:
                self.total_loss_hist.append(total_loss.detach().cpu())
        else:
            self.total_loss_hist.append(total_loss.detach().cpu())


        loss_dict = {}
        loss_dict["total_loss"] = total_loss
        loss_dict["loss_root"] = loss_root
        loss_dict["loss_vel"] = loss_vel
        loss_dict["loss_root_state"] = loss_root_state
        loss_dict["loss_vel_state"] = loss_vel_state
        loss_dict["torque_reg"] = torque_reg
        loss_dict["res_f_reg"] = res_f_reg
        loss_dict["foot_reg"] = foot_reg
        # print(self.target_ke)
        # print(self.target_kd)
        # print(self.body_mass)

        return loss_dict

    def backward(self, loss):
        loss.backward()

    def query(self, img_size=None):
        x_sims = []
        x_msms = []
        x_tsts = []  # target
        com_k = []
        data = {}

        # x_rest = trimesh.Trimesh(self.gtpoints[0]*10, self.faces,
        #        vertex_colors=self.colors, process=False)
        # in_bullet=False
        # use_urdf=False

        part_com = self.env.body_com.numpy()[..., None]
        part_mass = self.env.body_mass.numpy()
        x_rest = self.robot.urdf
        use_urdf = True
        for frame in range(len(self.obs)):
            obs = self.obs[frame]
            msm = self.msm[frame]
            tst = self.tstate[frame]
            grf = self.grfs[frame]
            jaf = self.jafs[frame]

            # get com (simulated)
            com = compute_com(obs, part_com, part_mass)
            com_k.append(compute_com(msm, part_com, part_mass))
            # x_msm = can2gym2gl(x_rest, msm, in_bullet=in_bullet, use_urdf=use_urdf, use_angle=True)

            x_msm = can2gym2gl(x_rest, msm, in_bullet=self.in_bullet, use_urdf=use_urdf)
            x_tst = can2gym2gl(x_rest, tst, in_bullet=self.in_bullet, use_urdf=use_urdf)
            x_sim = can2gym2gl(
                x_rest,
                obs,
                gforce=grf,
                com=com,
                in_bullet=self.in_bullet,
                use_urdf=use_urdf,
            )
            # x_sim = can2gym2gl(x_rest, obs, gforce=grf+jaf, in_bullet=self.in_bullet, use_urdf=use_urdf)

            x_sims.append(x_sim)
            x_msms.append(x_msm)
            x_tsts.append(x_tst)
        x_sims = np.stack(x_sims, 0)
        x_msms = np.stack(x_msms, 0)
        x_tsts = np.stack(x_tsts, 0)

        data["xs"] = x_sims
        data["xgt"] = x_msms
        data["tst"] = x_tsts
        data["com_k"] = com_k

        if img_size is not None:
            # get cameras: world to view = world to object + object to view
            # this triggers pyrender
            obj2world = self.target_q_vis[0]
            obj2world = se3_vec2mat(obj2world)
            world2obj = obj2world.inverse()

            obj2view = self.obj2view_vis[0]

            world2view = obj2view @ world2obj
            data["camera"] = world2view.cpu().numpy()
            data["camera"][:, 3] = self.ks_vis[0].cpu().numpy()
            data["img_size"] = img_size
        return data

    def save_network(self, epoch_label):
        if self.opts["local_rank"] == 0:
            save_dict = self.state_dict()
            param_path = "%s/params_%03d.pth" % (self.save_dir, epoch_label)
            torch.save(save_dict, param_path)

            return

    def load_network(self, model_path):
        states = torch.load(model_path, map_location="cpu")
        self.load_state_dict(states, strict=False)


class ForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rj_q, rj_qd, self):
        """
        rj_q:  T, bs, 7+B
        rj_qd: T, bs, 6+B / none
        """
        nfr, bs, ndof = rj_q.shape
        # save variables
        ctx.self = self
        ctx.bs = bs
        ctx.ndof = ndof
        ctx.rj_q = []
        ctx.rj_qd = []
        ctx.state_steps = []
        for i in self.frame2step:
            state = self.env.state(requires_grad=True)
            ctx.state_steps.append(state)

        rj_q = rj_q.view(nfr, -1)
        rj_qd = rj_qd.view(nfr, -1)
        for it, step in enumerate(self.frame2step):
            # q
            ctx.rj_q.append(wp.from_torch(rj_q[it]))
            # qd
            if rj_qd is None:
                rj_qd_sub = torch.cuda.FloatTensor(np.zeros(bs * (ndof - 1)))
            else:
                rj_qd_sub = rj_qd[it]
            rj_qd_sub = wp.from_torch(rj_qd_sub)
            ctx.rj_qd.append(rj_qd_sub)

        ctx.tape = wp.Tape()
        with ctx.tape:
            body_q = []
            body_qd = []
            msm = []
            for it, step in enumerate(self.frame2step):
                step = it
                eval_fk(
                    self.env, ctx.rj_q[it], ctx.rj_qd[it], None, ctx.state_steps[step]
                )  #
                body_q_sub = wp.to_torch(ctx.state_steps[step].body_q)  # bs*-1,7
                body_q_sub = body_q_sub.reshape(bs, -1, 7)
                body_q.append(body_q_sub)
                msm.append(body_q_sub.detach().cpu().numpy()[0])

                body_qd_sub = wp.to_torch(ctx.state_steps[step].body_qd)  # bs*-1,6
                body_qd_sub = body_qd_sub.reshape(bs, -1, 6)
                body_qd.append(body_qd_sub)
            body_q = torch.stack(body_q, 1)  # bs,T,dofs,7
            body_qd = torch.stack(body_qd, 1)
        return body_q, body_qd, msm

    @staticmethod
    def backward(ctx, adj_body_qs, adj_body_qd, _):
        self = ctx.self
        for it, step in enumerate(self.frame2step):
            step = it
            grad_body_q = adj_body_qs[:, it].reshape(-1, 7)  # bs, T, -1, 7
            ctx.state_steps[step].body_q.grad = wp.from_torch(
                grad_body_q, dtype=wp.transform
            )
            grad_body_qd = adj_body_qd[:, it].reshape(-1, 6)  # bs, T, -1, 7
            ctx.state_steps[step].body_qd.grad = wp.from_torch(
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
        else:
            rj_qd_grad = None

        if rj_q_grad.isnan().sum() > 0:
            pdb.set_trace()
        if rj_qd_grad.isnan().sum() > 0:
            pdb.set_trace()
        ctx.tape.zero()
        return (rj_q_grad, rj_qd_grad, None)


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
        ctx.body_mass = wp.from_torch(body_mass)

        # aux
        ctx.self = self

        # forward
        ctx.tape = wp.Tape()
        with ctx.tape:
            # TODO add kd/ke to optimization vars; not implemented by warp
            self.env.joint_target_kd = ctx.target_kd
            self.env.joint_target_ke = ctx.target_ke
            self.env.body_mass = ctx.body_mass

            # assign initial states
            eval_fk(self.env, ctx.q_init, ctx.qd_init, None, self.state_steps[0])

            # simulate
            self.grfs = []
            self.jafs = []
            for step in self.local_steps:
                self.state_steps[step].clear_forces()

                self.env.joint_target = ctx.refs[step]
                self.env.joint_act = ctx.torques[step]
                self.state_steps[step].body_f = ctx.res_f[step]

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
            obs = self.state_steps[0].body_q
            num_coords = obs.shape[0] // self.num_envs
            self.obs = [obs.numpy()[:num_coords]]
            wp_pos = [wp.to_torch(obs)]
            wp_vel = [wp.to_torch(self.state_steps[0].body_qd)]

            for step in self.frame2step[1:]:
                # for vis
                obs = self.state_steps[step + 1].body_q
                self.obs.append(obs.numpy()[:num_coords])
                wp_pos.append(wp.to_torch(obs))
                wp_vel.append(wp.to_torch(self.state_steps[step + 1].body_qd))
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
        self = ctx.self
        frame = 1
        for step in self.frame2step[1:]:
            # print('save to step: %d/from frame: %d'%(step, frame))
            self.state_steps[step + 1].body_q.grad = wp.from_torch(
                adj_body_qs[frame], dtype=wp.transform
            )
            self.state_steps[step + 1].body_qd.grad = wp.from_torch(
                adj_body_qd[frame], dtype=wp.spatial_vector
            )
            frame += 1
        # initial step
        self.state_steps[0].body_q.grad = wp.from_torch(
            adj_body_qs[0], dtype=wp.transform
        )
        self.state_steps[0].body_qd.grad = wp.from_torch(
            adj_body_qd[0], dtype=wp.spatial_vector
        )

        # return adjoint w.r.t. inputs
        ctx.tape.backward()

        grad = [wp.to_torch(v) for k, v in ctx.tape.gradients.items()]
        print("max grad:")
        print(torch.cat([i.reshape(-1) for i in grad]).abs().max())

        try:
            q_init_grad = wp.to_torch(ctx.tape.gradients[ctx.q_init]).clone()
            remove_nan(q_init_grad, self.num_envs)
        except:
            q_init_grad = None

        try:
            qd_init_grad = wp.to_torch(ctx.tape.gradients[ctx.qd_init]).clone()
            remove_nan(qd_init_grad, self.num_envs)
        except:
            qd_init_grad = None

        refs_grad = [
            wp.to_torch(ctx.tape.gradients[i]) for i in ctx.refs if i.requires_grad
        ]
        if len(refs_grad) > 0:
            refs_grad = torch.stack(refs_grad, 0).clone()
            remove_nan(refs_grad, self.num_envs)
        else:
            refs_grad = None

        torques_grad = [wp.to_torch(ctx.tape.gradients[i]) for i in ctx.torques]
        if len(torques_grad) > 0:
            torques_grad = torch.stack(torques_grad, 0).clone()
            remove_nan(torques_grad, self.num_envs)
        else:
            torques_grad = None

        res_f_grad = [wp.to_torch(ctx.tape.gradients[i]) for i in ctx.res_f]
        if len(res_f_grad) > 0:
            res_f_grad = torch.stack(res_f_grad, 0).clone()
            remove_nan(res_f_grad, self.num_envs)
        else:
            res_f_grad = None

        try:
            target_kd_grad = wp.to_torch(ctx.tape.gradients[ctx.target_kd]).clone()
            target_ke_grad = wp.to_torch(ctx.tape.gradients[ctx.target_ke]).clone()
            remove_nan(target_ke_grad, self.num_envs)
            remove_nan(target_kd_grad, self.num_envs)
        except:
            target_kd_grad = None
            target_ke_grad = None

        try:
            body_mass_grad = wp.to_torch(ctx.tape.gradients[ctx.body_mass]).clone()
            remove_nan(body_mass_grad, self.num_envs)
        except:
            body_mass_grad = None

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
            None,
        )


