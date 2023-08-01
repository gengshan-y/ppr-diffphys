import os
import sys
import copy
import time
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from pytorch3d import transforms
import pdb
import glob
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
import trimesh

from utils.io import vis_kps
from env_utils.torch_utils import Embedding, NeRF, clip_grad
from nnutils.nerf import SkelHead, FrameCode, RTExpMLP
from nnutils.robot import URDFRobot
from nnutils.urdf_utils import articulate_robot_rbrt, articulate_robot_rbrt_batch,\
                               articulate_robot
from nnutils.geom_utils import se3_vec2mat, se3_mat2vec, rot_angle, vec_to_sim3, \
                                create_base_se3, refine_rt, fid_reindex
from env_utils.dataloader import parse_amp
from env_utils.import_urdf import parse_urdf
from env_utils.articulation import eval_fk
from env_utils.integrator_euler import SemiImplicitIntegrator

import cv2
import warp as wp
import warp.sim
wp.init()

def zero_grad_list(paramlist):
    """
    Clears the gradients of all optimized :class:`torch.Tensor`
    """
    for p in paramlist:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    
def rotate_frame(global_q, target_q):
    """
    global_q: 7
    target_q: bs,t,7
    """
    # root states: T = Tg @ T_t, bs,T
    global_qmat = se3_vec2mat(global_q)
    if len(global_q.shape)==1:
        global_qmat = global_qmat[None, None]
    target_qmat = se3_vec2mat(target_q)
    target_qmat = global_qmat @ target_qmat
    target_q = se3_mat2vec( target_qmat, outdim = target_q.shape[-1])
    return target_q
    
def rotate_frame_vel(global_q, target_qd):
    # only rotate the first 3 elements
    global_qd = global_q.clone()
    global_qd[...,:3] = 0
    target_qd_rev = torch.cat([target_qd[...,3:], target_qd[...,:3]],-1)
    rot = rotate_frame(global_qd, target_qd)[...,:3]
    trn = rotate_frame(global_qd, target_qd_rev)[...,:3] 
    target_qd_rt = torch.cat([rot, trn],-1)
    return target_qd_rt

        
def state_to_msm(state_q, state_qd, foot_pos, in_bullet):
    msm = {}
    msm['pos'] = state_q[0:3]  .detach().cpu().numpy() 
    msm['orn'] = state_q[3:7]  .detach().cpu().numpy() 
    msm['vel'] = state_qd[0:3] .detach().cpu().numpy() 
    msm['avel']= state_qd[3:6] .detach().cpu().numpy() 
    msm['jang']= state_q[7:19] .detach().cpu().numpy() 
    msm['jvel']= state_qd[6:18].detach().cpu().numpy() 
    msm['kp'] = foot_pos.view(-1,3) .detach().cpu().numpy()
    msm['kp_vel'] = msm['kp']*0
    gl2bullet(msm, in_bullet)
    msm['kp'] = msm['kp'].flatten()
    msm['kp_vel'] = msm['kp_vel'].flatten()
    return msm
            
def compute_com(body_q, part_com, part_mass):
    body_com = R.from_quat(body_q[:,3:]).as_matrix() @ part_com
    body_com = body_com[...,0] + body_q[:,:3]
    com = (body_com * part_mass[:,None]).sum(0) / part_mass.sum()
    return com
        
def clip_loss(loss_seq, th):
    """
    bs,T
    """
    #if th==0:
    #    th=loss_seq.median()*10
    #clip_val,clip_idx = torch.max(loss_seq>th,1)
    for i in range(len(loss_seq)):
        if th==0:
            loss_sub = loss_seq[i]
            th=loss_sub[loss_sub>0].median()*10
        clip_val,clip_idx = torch.max(loss_seq[i]>th,0)
        if clip_val==1:
            loss_seq[i,clip_idx:] = 0
        #if clip_val[i]==1:
        #    loss_seq[i,clip_idx[i]:] = 0
    if loss_seq.sum()>0:
        loss_seq = loss_seq[loss_seq>0].mean()
    else:
        loss_seq = loss_seq.mean()
    return loss_seq

def rectify_func(pred_est_q, rotate_frame, global_q, to_quat=True):
    def rectify_func_rt(steps_fr):
        pred_q = pred_est_q(steps_fr)
        pred_q = rotate_frame(global_q, pred_q)
        if not to_quat:
            rot = transforms.quaternion_to_axis_angle(pred_q[...,3:])
            pred_q = torch.cat( [pred_q[...,:3], rot], -1)
        return pred_q
    return rectify_func_rt

def compose_func(delta_root_mlp, compose_delta, target_q):
    def compose_func_rt(steps_fr):
        bs,nstep = steps_fr.shape
        delta_root = delta_root_mlp(steps_fr.reshape(-1,1))
        delta_root = delta_root.view(bs,nstep,-1)
        est_q = compose_delta(target_q, delta_root)
        est_q = transforms.quaternion_to_axis_angle(est_q)
        return est_q
    
    return compose_func_rt
        
def se3_loss(pred, gt,rot_ratio=0.1):
    """
    ...,7
    """
    # find nan values
    nanid = torch.logical_or( pred.sum(-1).isnan(), gt.sum(-1).isnan() )

    trn_loss = (pred[...,:3] - gt[...,:3]).pow(2).sum(-1)

    rot_pred = pred[...,3:]
    rot_gt = gt[...,3:]
    if rot_pred.shape[-1]==3:
        rot_pred = transforms.axis_angle_to_matrix(rot_pred)
        rot_gt = transforms.axis_angle_to_matrix(rot_gt) 
        rot_gti = rot_gt.inverse()
    elif rot_pred.shape[-1]==4:
        rot_pred = transforms.quaternion_to_matrix(rot_pred[...,[3,0,1,2]]) # xyzw => wxyz
        rot_gti = transforms.quaternion_invert(rot_gt[...,[3,0,1,2]])
        rot_gti = transforms.quaternion_to_matrix(rot_gti) 
    rot_loss = rot_angle(rot_pred @ rot_gti)

    #loss = trn_loss
    loss = 0.1*trn_loss + 0.1*rot_loss*rot_ratio
    #loss = 0.1*trn_loss + 0.01*rot_loss
    #loss = trn_loss + 0.01*rot_loss
    loss[nanid] = 0
    return loss

def gl2bullet(msm, in_bullet):
    ndim = msm['pos'].ndim-1
    gl_to_issac = np.asarray([[0,0,1], [1,0,0], [0,1,0]]).reshape(ndim*(1,)+(3,3))
    msm['pos'] =         (gl_to_issac @ msm['pos'][...,None])   [...,0]
    if in_bullet:
        shape=msm['orn'].shape[:-1]
        orn = R.from_quat(msm['orn'].reshape((-1,4))).as_matrix() # N,3,3
        msm['orn'] = R.from_matrix(orn @ gl_to_issac.reshape((-1,3,3))).as_quat()
        msm['orn']  = msm['orn'].reshape(shape+(4,))
    msm['orn'][...,:3] = (gl_to_issac @ msm['orn'][...,:3,None])[...,0] # xyzw

    msm['vel']  = (gl_to_issac @ msm['vel'][...,None])    [...,0]
    msm['avel'] = (gl_to_issac @ msm['avel'][...,None])   [...,0]
    if 'kp' in msm.keys():
        msm['kp'] =     (gl_to_issac @ msm['kp'][...,None])    [...,0]
        msm['kp_vel'] = (gl_to_issac @ msm['kp_vel'][...,None])[...,0]


def bullet2gl(msm, in_bullet):
    # in_bullet: convert the rest mesh as well (for those in bullet)
    ndim = msm['pos'].ndim-1
    issac_to_gl = np.asarray([[0,1,0], [0,0,1], [1,0,0]]).reshape(ndim*(1,)+(3,3))
    msm['pos'] =         (issac_to_gl @ msm['pos'][...,None])   [...,0]
    if in_bullet:
        shape=msm['orn'].shape[:-1]
        orn = R.from_quat(msm['orn'].reshape((-1,4))).as_matrix() # N,3,3
        msm['orn'] = R.from_matrix(orn @ issac_to_gl.reshape((-1,3,3))).as_quat()
        msm['orn']  = msm['orn'].reshape(shape+(4,))
    msm['orn'][...,:3] = (issac_to_gl @ msm['orn'][...,:3,None])[...,0] # xyzw

    msm['vel']  = (issac_to_gl @ msm['vel'][...,None])    [...,0]
    msm['avel'] = (issac_to_gl @ msm['avel'][...,None])   [...,0]

def pred_est_q(steps_fr, nerf_root_rts, nerf_body_rts, bg_rts):
    """
    bs,T
    robot2world = scale(bg2world @ bg2view^-1 @ root2view @ se3)
    """
    bs,n_fr = steps_fr.shape
    device = steps_fr.device
    data_offset = bg_rts.data_offset
    vidid,_ = fid_reindex(steps_fr, \
                        len(data_offset)-1, data_offset)
    vidid = vidid.long()

    # canonical to view
    rt_base = create_base_se3(bs*n_fr, device)
    pred_mat = nerf_root_rts(steps_fr.reshape(-1,1))
    pred_mat = refine_rt(rt_base, pred_mat) # -1,3,4

    ##TODO further rotate along x axis
    #rot_offset = cv2.Rodrigues(np.asarray([0.2,0.,0.]))[0]
    #rot_offset = torch.Tensor(rot_offset).to(device)[None]
    #pred_mat[:,:3,:3] = rot_offset @ pred_mat[:,:3,:3]

    # sim3 robot transform 
    sim3 = nerf_body_rts.compute_sim3(vidid[:,0])

    # TODO do not update sim3 and root rts
    #sim3 = sim3.detach()
    #pred_mat = pred_mat.detach() 

    center, orient, scale = vec_to_sim3(sim3)
    se3 = torch.cat([orient, center[...,None]],2)
    zero_ones = torch.zeros_like(se3[:,:1])
    zero_ones[...,-1] = 1
    se3 = torch.cat([se3, zero_ones],1)

    zero_ones = torch.zeros_like(pred_mat[:,:1])
    zero_ones[...,-1] = 1
    pred_mat = torch.cat([pred_mat, zero_ones], 1) # -1,4,4
    pred_mat = pred_mat.view(bs,n_fr,4,4) @ se3[:,None]
    pred_mat = pred_mat.view(-1,4,4)

    # intermediate outputs
    obj2view = pred_mat.clone().view(bs,n_fr,4,4)
    obj2view[...,:3,3] /= scale.mean(-1).view(bs,1,1) # TODO use ave scale

    # background transform
    bgrt = bg_rts.get_rts(steps_fr.reshape(-1)) # -1,3,4
    bgrt = torch.cat([bgrt, zero_ones], 1) # -1,4,4
    pred_mat = bgrt.inverse() @ pred_mat
    bg2world = se3_vec2mat(bg_rts.bg2world[vidid.view(-1)])
    bg2world[...,:3,3] *= bg_rts.bg2fg_scale[vidid.view(-1)][...,None].exp()
    pred_mat = bg2world @ pred_mat
    
    # cv to gl coords
    cv2gl = torch.eye(4).to(pred_mat.device)
    cv2gl[1,1] =-1
    cv2gl[2,2] =-1
    pred_mat = cv2gl[None] @ pred_mat
    pred_mat[...,:3,3] /= scale.mean(-1)[:,None].repeat(1,n_fr).view(-1,1) # TODO use ave scale

    pred_q = se3_mat2vec( pred_mat ) # xyzw
    pred_q = pred_q.view(bs,n_fr,-1)
    return pred_q, obj2view

def pred_est_ja(steps_fr, nerf_body_rts, env):
    """
    bs,T
    """
    bs,n_fr = steps_fr.shape
    device = steps_fr.device
    data_offset = nerf_body_rts.data_offset
    vidid,_ = fid_reindex(steps_fr, \
                        len(data_offset)-1, data_offset)
    vidid = vidid.long()

    # pred joint angles
    _,pred_joints = nerf_body_rts.forward_abs(steps_fr.reshape(-1,1))
    pred_joints = pred_joints.view(bs,n_fr,-1)

    # update joint locations
    #TODO rightnow, assume no extra scaling in the simulation space
    joint_origin = np.stack([i.origin for i in nerf_body_rts.urdf.joints],0)
    joint_origin = torch.Tensor(joint_origin).to(device)[None].repeat(bs,1,1,1)
    jlen_scale = nerf_body_rts.compute_jlen_scale(vidid[:,0])
    joint_tmat = nerf_body_rts.update_joints(nerf_body_rts.urdf, nerf_body_rts.joints, jlen_scale)
    map_idx = np.asarray(nerf_body_rts.urdf.unique_body_idx[1:]) - 3
    joint_origin[:,map_idx,:3,3] = joint_tmat
    joint_origin = joint_origin[:,map_idx] #TODO get only the unique ones

    # move to warp skeleton
    joint_quat = transforms.matrix_to_quaternion(joint_origin[...,:3,:3]) # for a single video
    joint_tmat = joint_origin[...,:3,3]
    joint_origin = torch.cat([joint_tmat, joint_quat[...,[1,2,3,0]]],-1)
    # set first joitn to identity
    zero_ones = torch.zeros_like(joint_origin[:,:1]); zero_ones[:,0,-1] = 1
    joint_origin = torch.cat([zero_ones, joint_origin], 1)
    joint_origin = joint_origin.view(-1,7)
    env.joint_X_p = wp.from_torch(joint_origin, dtype=wp.transform)
    return pred_joints

class WarpRootMLP(nn.Module):
    def __init__(self, banmo_root_mlp, banmo_body_mlp, banmo_bg_mlp):
        super(WarpRootMLP, self).__init__()
        self.root_mlp = copy.deepcopy(banmo_root_mlp)
        #self.body_mlp = copy.deepcopy(banmo_body_mlp)
        self.body_mlp = banmo_body_mlp.mlp
        self.bg_mlp = copy.deepcopy(banmo_bg_mlp)
        #self.bg_mlp.ft_bgcam=True # asssuming it's accurate enough

    def forward(self, x):
        out,_ = pred_est_q(x, self.root_mlp, self.body_mlp, self.bg_mlp)
        return out

    def override_states(self, banmo_root_mlp, banmo_body_mlp, banmo_bg_mlp):
        self.root_mlp.load_state_dict(banmo_root_mlp.state_dict())
        #self.body_mlp.load_state_dict(banmo_body_mlp.state_dict())
        self.bg_mlp.load_state_dict(banmo_bg_mlp.state_dict())
    
    def override_states_inv(self, banmo_root_mlp, banmo_body_mlp, banmo_bg_mlp):
        banmo_root_mlp.load_state_dict(self.root_mlp.state_dict())
        #banmo_body_mlp.load_state_dict(self.body_mlp.state_dict())
        banmo_bg_mlp.load_state_dict(self.bg_mlp.state_dict())

class WarpBodyMLP(nn.Module):
    def __init__(self, banmo_mlp):
        super(WarpBodyMLP, self).__init__()
        self.mlp = copy.deepcopy(banmo_mlp)

    def forward(self, x):
        _,out = self.mlp.forward_abs(x)
        return out
    
    def override_states(self, banmo_mlp):
        self.mlp.load_state_dict(banmo_mlp.state_dict())
    
    def override_states_inv(self, banmo_mlp):
        banmo_mlp.load_state_dict(self.mlp.state_dict())

class Scene(nn.Module):
    def __init__(self, opts, dataloader, dt = 5e-4, use_dr = False, device='cuda'):
        super(Scene, self).__init__()
        #dt = dt/10
        #dt = dt/5
        dt = dt/2
        self.opts = opts
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.logname)
        self.dt = dt
        self.device = device
        if use_dr:
            self.preset_data_dr(dataloader)
        else:
            self.preset_data(dataloader)
        self.use_dr = use_dr
        
        # mlp
        if opts.pre_skel=="a1":
            urdf_path='mesh_material/a1/urdf/a1.urdf'
            in_bullet=True
            kp=220.
            kd=2.
            shape_ke=1.e+4
            shape_kd=0
        elif opts.pre_skel=="wolf":
            urdf_path='mesh_material/wolf.urdf'
            in_bullet=False
            self.joint_attach_ke = 32000.
            self.joint_attach_kd = 100.
            kp=2200.
            kd=2.
            shape_ke=1000
            shape_kd=100
        elif opts.pre_skel=="wolf_mod":
            urdf_path='mesh_material/wolf_mod.urdf'
            in_bullet=False
            self.joint_attach_ke = 8000.
            self.joint_attach_kd = 200.
            kp=660.
            kd=5.
            shape_ke=1000
            shape_kd=100
            #self.joint_attach_ke = 16000.
            #self.joint_attach_kd = 100.
            #kp=220.
            #kd=2.
        elif opts.pre_skel=="laikago":
            urdf_path='mesh_material/laikago/laikago.urdf'
            in_bullet=False
            #urdf_path='utils/tds/data/laikago/laikago_mod.urdf'
            #in_bullet=True
            self.joint_attach_ke = 16000.
            self.joint_attach_kd = 200.
            kp=220.
            kd=2.
            shape_ke=1.e+4
            shape_kd=0
            if opts.rollout:
                urdf_path='utils/tds/data/laikago/laikago_toes_zup_joint_order.urdf'
                in_bullet=True
        elif opts.pre_skel=="human":
            urdf_path='mesh_material/human.urdf'
            in_bullet=False
            self.joint_attach_ke = 64000.
            self.joint_attach_kd = 150. # tune this such that it would not blow up
            kp=20.
            kd=2.
            shape_ke=1000
            shape_kd=100
        elif opts.pre_skel=="human_mod":
            urdf_path='mesh_material/human_mod.urdf'
            in_bullet=False
            self.joint_attach_ke = 8000.
            self.joint_attach_kd = 200.
            kp=660.
            kd=5.
            shape_ke=1000
            shape_kd=100
            #kp=20.
            #kd=2.
        elif opts.pre_skel=="human_amp":
            urdf_path='mesh_material/human_amp.urdf'
            in_bullet=False
            kp=20.
            kd=2.
            shape_ke=1000
            shape_kd=100
        self.in_bullet=in_bullet
        self.robot = URDFRobot(urdf_path=urdf_path)

        ## pose code
        #self.pose_code = FrameCode(num_freq = opts.num_freq, 
        #                           embedding_dim = opts.t_embed_dim, 
        #                           vid_offset = self.data_offset)
        #self.rest_pose_code = nn.Embedding(1, opts.t_embed_dim)
        #self.nerf_root_rts = RTExpMLP(self.gt_steps,
        #                     opts.num_freq,opts.t_embed_dim, self.data_offset,
        #                     delta=False)
        #self.nerf_body_rts = SkelHead(urdf=self.robot.urdf,joints=self.robot.joints,
        #      sim3=self.robot.sim3, rest_angles=self.robot.rest_angles,
        #                pose_code=self.pose_code,
        #                rest_pose_code=self.rest_pose_code,
        #                data_offset=self.data_offset,
        #                in_channels=opts.t_embed_dim,
        #                out_channels=self.robot.num_dofs)

        ## load weights
        #states = torch.load('tmp/laikago.pth', map_location='cpu')
        #body_states = self.rm_module_prefix(states,
        #    prefix='module.nerf_body_rts')
        #self.nerf_body_rts.load_state_dict(body_states, strict=True)

        # env
        self.articulation_builder = wp.sim.ModelBuilder()
        #TODO change path
        parse_urdf(urdf_path,
            self.articulation_builder,
            xform=wp.transform(np.array((0.0, 0.417, 0.0)), wp.quat_from_axis_angle((1.0, 0.0, 0.0), 0.0)),
            floating=True,
            density=1000, # collision geometry needs >0 density, but this will ignore urdf mass
            armature=0.01, # additional inertia
            stiffness=220., # ke gain
            damping=2., # kd gain but was set to zero somhow
            shape_ke=shape_ke, # collsion spring/damp/friction
            shape_kd=shape_kd,
            shape_kf=1.e+2,
            shape_mu=1, # use a large value to make ground sticky
            limit_ke=0, # useful when joints violating limits
            limit_kd=0)
            #limit_ke=1.e+4, # useful when joints violating limits
            #limit_kd=1.e+1)

        # make feet heavier
        name2link_idx = [(link.name,it) for it,link in enumerate(self.robot.urdf.links)]
        dict_unique_body = dict(enumerate(self.robot.urdf.unique_body_idx))
        self.dict_unique_body_inv = {v: k for k, v in dict_unique_body.items()}
        name2link_idx = [(link.name,self.dict_unique_body_inv[it]) for it,link \
          in enumerate(self.robot.urdf.links) if it in dict_unique_body.values()]
        name2link_idx = dict(name2link_idx)
        # lighter trunk
        #for kp_name in ['link_136_Bauch_Y', 'link_137_Bauch.001_Y', 
        #                'link_138_Brust_Y', 'link_139_Hals_Y']:
        #    kp_idx = name2link_idx[kp_name]
        #    self.articulation_builder.body_mass[kp_idx] *= 1./10

        if hasattr(self.robot.urdf, 'kp_links'):
            # for human and wolf
            for i in range(len(self.articulation_builder.body_mass)):
                self.articulation_builder.body_mass[i] = 2
            for kp_name in self.robot.urdf.kp_links:
                kp_idx = name2link_idx[kp_name]
                self.articulation_builder.body_mass[kp_idx] *= 2
                tup = self.articulation_builder.shape_geo_scale[kp_idx]
                self.articulation_builder.shape_geo_scale[kp_idx] = (tup[0]*2, 
                                                                    tup[1]*2, tup[2]*2)

        self.n_dof= len(self.articulation_builder.joint_q)-7
        self.n_links= len(self.articulation_builder.body_q)
        self.articulation_builder.joint_target_ke = [0.0] * 6 + [kp] * \
                                    (len(self.articulation_builder.joint_target_ke)-6)
        self.articulation_builder.joint_target_kd = [0.0] * 6 + [kd] * \
                                    (len(self.articulation_builder.joint_target_ke)-6)

        # integrator
        self.integrator = SemiImplicitIntegrator()

        # torch parameters
        if self.use_dr:
            self.global_q = nn.Parameter( torch.cuda.FloatTensor([0.,0.,0.,0.,0.,0.,1.]) )
        else:
            self.global_q = nn.Parameter( torch.cuda.FloatTensor([0.,-0.03,0.,0.,0.,0.,1.]) )
        #self.global_q = nn.Parameter( torch.cuda.FloatTensor([0.,-0.042,0.,0.,0.,0.,1.]) )

        self.target_ke = nn.Parameter( torch.cuda.FloatTensor(self.articulation_builder.joint_target_ke))
        self.target_kd = nn.Parameter( torch.cuda.FloatTensor(self.articulation_builder.joint_target_kd))
        self.body_mass = nn.Parameter( torch.cuda.FloatTensor(self.articulation_builder.body_mass))

        #TODO
        self.torque_mlp = NeRF(tscale = 1./self.gt_steps, N_freqs = 6, 
                        D=8, W=256,
                        out_channels=self.n_dof,
                        in_channels_xyz=13,
                        )
        
        self.residual_f_mlp = NeRF(tscale = 1./self.gt_steps, N_freqs = 6, 
                        D=8, W=256,
                        out_channels=6*self.n_links,
                        in_channels_xyz=13,
                        )
        
        self.delta_root_mlp = NeRF(tscale = 1./self.gt_steps, N_freqs = 6, 
                        D=8, W=256,
                        out_channels=6,
                        in_channels_xyz=13,
                        )
        
        self.vel_mlp = NeRF(tscale = 1./self.gt_steps, N_freqs = 6, 
                        D=8, W=256,
                        out_channels=6+self.n_dof,
                        in_channels_xyz=13,
                        )
        
        self.delta_joint_ref_mlp = NeRF(tscale = 1./self.gt_steps, N_freqs = 6, 
                        D=8, W=256,
                        out_channels=self.n_dof,
                        in_channels_xyz=13,
                        )
        
        self.delta_joint_est_mlp = NeRF(tscale = 1./self.gt_steps, N_freqs = 6, 
                        D=8, W=256,
                        out_channels=self.n_dof,
                        in_channels_xyz=13,
                        )
        
        if self.use_dr:
            #TODO create new modules
            self.delta_joint_est_mlp = WarpBodyMLP(self.nerf_body_rts)
            self.delta_root_mlp = WarpRootMLP(self.nerf_root_rts, self.delta_joint_est_mlp, self.bg_rts)
            #self.delta_joint_ref_mlp = WarpBodyMLP(self.nerf_body_rts)

        # optimizer
        self.add_optimizer(opts)
        self.total_loss_hist = []
        
        # functions
        self.pred_est_q_rect = rectify_func(self.pred_est_q, rotate_frame, self.global_q)
        self.pred_est_q_rect_rod = rectify_func(self.pred_est_q, rotate_frame, self.global_q, to_quat=False)

        # mpc
        if opts.rollout:
            os.sys.path.insert(0, 'utils/tds')
            import laikago_tds_sim_clean as robot_sim
            from utils.tds_env import _setup_hybrid_controller, _setup_controller, _setup_pd_controller
            self.sim_robot = robot_sim.SimpleRobot(simulation_time_step=self.dt)
            #self.controller = _setup_pd_controller(self.sim_robot)
            self.controller = _setup_hybrid_controller(self.sim_robot)
            #self.controller = _setup_controller(self.sim_robot)

        # other hypoter parameters
        self.th_multip = 0 # seems to cause problems

    def pred_est_q(self,steps_fr):
        out,obj2view = pred_est_q(steps_fr, self.nerf_root_rts, self.nerf_body_rts, self.bg_rts)
        return out, obj2view

    def pred_est_ja(self,steps_fr):
        out = pred_est_ja(steps_fr, self.nerf_body_rts, self.env)
        return out

    def warmup_state_estimate(self):
        # warmup mlp for reference trajectories
        for it in range(1000):
        #for it in range(10):
            # get a batch of clips
            steps_fr = torch.cuda.LongTensor(range(self.gt_steps))[None] # frames
            target_q, target_ja, _,_ = \
                                self.get_batch_input(self.amp_info_func,steps_fr,self.in_bullet)
            # inference
            pred_q      = self.pred_est_q(steps_fr)
            pred_joints = self.pred_est_ja(steps_fr)

            # compare with gts
            loss_joints = (target_ja - pred_joints).pow(2).mean()
            loss_q = se3_loss(target_q, pred_q).mean()

            total_loss = loss_joints + loss_q
            
            # train
            self.backward([total_loss]) 
            grad_list = self.update()
            if it%100==0:
                print('step: %s/loss: %.4f'%(it, total_loss))


    @staticmethod
    def rm_module_prefix(states, prefix='module'):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[:len(prefix)] == prefix:
                i = i[len(prefix)+1:]
                new_dict[i] = v
        return new_dict
        
    def reinit_envs(self,num_envs, wdw_length, is_eval=False, overwrite=False):
        self.num_envs = num_envs
        self.wdw_length = wdw_length # frames
        self.local_steps = range(self.skip_factor * self.wdw_length)
        self.local_steps_fr = torch.cuda.LongTensor(self.local_steps) / self.skip_factor # frames
        self.wdw_length_full = len(self.local_steps)
        
        self.frame2step = [0]
        for i in range(len(self.local_steps)+1):
            if (i+1)%self.skip_factor==0:
                self.frame2step.append(i)

        if is_eval:
            env_name='eval_env'
            state_name='eval_state'
        else:
            env_name='train_env'
            state_name='train_state'
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
            for i in range(len(self.local_steps)+1):
                state = self.env.state(requires_grad=True)
                self.state_steps.append(state)

#            if not self.use_dr:
            self.env.collide(self.state_steps[0]) # ground contact, call it once
            
            setattr(self, env_name, self.env)
            setattr(self, state_name, self.state_steps)
        
    def preset_data_dr(self, model_dict):
        self.bg_rts = model_dict['bg_rts']
        self.nerf_root_rts = model_dict['nerf_root_rts']
        self.nerf_body_rts = model_dict['nerf_body_rts']
        self.ks_params = model_dict['ks_params']

        self.data_offset = self.nerf_body_rts.data_offset
        self.samp_int = 0.1
        self.gt_steps = self.data_offset[-1]-1
        self.gt_steps_visible = self.gt_steps
        self.max_steps = int(self.samp_int * self.gt_steps / self.dt)
        self.skip_factor = self.max_steps // self.gt_steps

    def preset_data(self, dataloader):
        if hasattr(dataloader, 'amp_info'):
            amp_info = dataloader.amp_info

        self.data_offset = dataloader.data_info['offset']
        self.samp_int = dataloader.samp_int

        self.gt_steps = len(amp_info)-1 # for k frames, need to step k-1 times
        self.gt_steps_visible = self.gt_steps
        self.max_steps = int(self.samp_int * self.gt_steps / self.dt)
        self.skip_factor = self.max_steps // self.gt_steps
        
        # data query
        self.amp_info_func = scipy.interpolate.interp1d(
            np.asarray(range(self.gt_steps+1)),
            amp_info,
            kind="linear",
            fill_value="extrapolate",
            axis=0)

    def add_params_to_dict(self, clip_grad=False):
        opts = self.opts
        params_list = []
        lr_dict = {}
        grad_dict = {}
        is_invalid_grad=False
        for name,p in self.named_parameters():
            params = {'params': [p]}
            params_list.append(params)
            if 'bg2fg_scale' in name:
                #print('bg2fg_scale')
                #print(p)
                lr = 100*opts.learning_rate # explicit variables
                g_th = 0.1
            elif 'bg2world' in name:
                lr = 0*opts.learning_rate # do not update
                g_th = 0
            elif 'global_q' in name:
                lr = opts.learning_rate
                g_th = 100
            elif 'target_ke' == name or 'target_kd' == name:
                lr = 20*opts.learning_rate
                g_th = 0.1
            elif 'attach_ke' == name or 'attach_kd' == name:
                lr = 20*opts.learning_rate
                g_th = 0.1
            elif 'body_mass' == name:
                lr = 20*opts.learning_rate
                g_th = 0.1
            elif 'torque_mlp' in name:
                lr = opts.learning_rate
                g_th = 1
            elif 'residual_f_mlp' in name:
                lr = opts.learning_rate
                g_th = 1
            elif 'delta_root_mlp' in name:
                lr = opts.learning_rate
                g_th = 1
            elif 'vel_mlp' in name:
                lr = opts.learning_rate
                g_th = 1
            elif 'delta_joint_est_mlp' in name:
                lr = opts.learning_rate
                g_th = 1
            elif 'delta_joint_ref_mlp' in name:
                lr = opts.learning_rate
                g_th = 1
            else:
                lr = opts.learning_rate
                g_th = 1
            lr_dict[name] = lr
            if clip_grad:
                try:
                    pgrad_nan = p.grad.isnan()
                    if pgrad_nan.sum()>0:
                        print('%s grad invalid'%name)
                        is_invalid_grad=True
                except: pass
                grad_dict['grad_'+name] = clip_grad_norm_(p, g_th)

        return params_list, lr_dict, grad_dict, is_invalid_grad
    
    def clip_grad(self):
        """
        gradient clipping
        """
        _,_,grad_dict,is_invalid_grad = self.add_params_to_dict(clip_grad=True)
    
        if is_invalid_grad:
            zero_grad_list(self.parameters())
    
        return grad_dict

    def add_optimizer(self, opts):
        params_list, lr_dict, _,_ = self.add_params_to_dict()
        for (name, lr) in lr_dict.items():
            print('optimized params: %.5f/%s'%(lr, name))

        self.optimizer = torch.optim.AdamW(
        params_list,
        lr=opts.learning_rate)
        #self.optimizer = torch.optim.SGD(
        #params_list,
        #lr=1)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
        list(lr_dict.values()),
        int(opts.num_epochs * 200),
        pct_start=0.02, # use 2%
        cycle_momentum=False,
        anneal_strategy='linear',
        final_div_factor=1./5, div_factor = 25,
        )
        #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
        #list(lr_dict.values()),
        #int(1e6), # 1000k steps
        #pct_start=0.02, # use 2%
        #cycle_momentum=False,
        #anneal_strategy='linear',
        #final_div_factor=1., div_factor = 1.,
        #)

    def update(self):
        grad_dict = self.clip_grad()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_dict
    
    def sample_sys_state(self, steps_fr):
        batch = {}
        batch['target_q'], batch['obj2view']  = self.pred_est_q(steps_fr)
        batch['target_ja'] = self.pred_est_ja(steps_fr)
        #TODO this is problematic due to the discrete root pose
        #batch['target_qd'] = self.compute_gradient(self.pred_est_q,  steps_fr.clone()) / self.samp_int 
        #batch['target_jad']= self.compute_gradient(self.pred_est_ja, steps_fr.clone()) / self.samp_int
        batch['target_qd'] = torch.zeros_like(batch['target_q'])[...,:6]
        batch['target_jad']= torch.zeros_like(batch['target_ja'])
        # ks
        vidid,_ = fid_reindex(steps_fr, len(self.data_offset)-1, self.data_offset)
        batch['ks'] = self.ks_params[vidid.long()]

        return batch
    
    @staticmethod
    def get_batch_input(amp_info_func, steps_fr, in_bullet):
        """
        steps_fr: bs, T
        target_q:    bs,T,7
        target_ja:  bs,T,dof
        """
        # pos/orn/ref joints from data
        amp_info = amp_info_func(steps_fr.cpu().numpy())
        msm = parse_amp(amp_info)
        bullet2gl(msm, in_bullet)
        target_ja  = torch.cuda.FloatTensor(msm['jang'])
        target_pos = torch.cuda.FloatTensor(msm['pos'])
        target_orn = torch.cuda.FloatTensor(msm['orn'])
        target_jad = torch.cuda.FloatTensor(msm['jvel'])
        target_vel = torch.cuda.FloatTensor(msm['vel'])
        target_avel= torch.cuda.FloatTensor(msm['avel'])

        target_q = torch.cat([ target_pos[...,:], target_orn[...,:] ], -1) # bs, T, 7
        target_qd = torch.cat([ target_vel[...,:], target_avel[...,:] ], -1) # bs, T, 6
        return target_q, target_ja, target_qd, target_jad
        
    @staticmethod
    def compose_delta(target_q, delta_root):
        """
        target_q: bs, T, 7
        delta_root: bs, T, 6
        """
        delta_qmat = se3_vec2mat(delta_root)
        target_qmat = se3_vec2mat(target_q)
        target_qmat = delta_qmat @ target_qmat
        target_q = se3_mat2vec( target_qmat )
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
            y_sub = y [...,i:i+1]
            d_output = torch.ones_like(y_sub, requires_grad=False, device=y.device)
            gradient = torch.autograd.grad(
                outputs=y_sub,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            gradients.append( gradient[...,None] )
        gradients = torch.cat(gradients,-1) # ...,input-dim, output-dim
        return gradients


    def get_net_pred(self,steps_fr):
        """
        steps_fr: bs,T
        torques:  bs,T,dof
        delta_ja, bs,T,dof
        """
        # additional torques from net
        bs,nstep = steps_fr.shape
        torques = self.torque_mlp(steps_fr.reshape(-1,1))
        torques = torch.cat([torch.zeros_like(torques[:,:1].repeat(1,6)), torques],1)
        torques = torques.view(bs,nstep,-1)
        #torques *= 0
        
        # residual force
        res_f = self.residual_f_mlp(steps_fr.reshape(-1,1))
        res_f = res_f.view(bs,nstep,-1,6)
        #res_f[:,:,1:] = 0
        res_f[...,3:6] *= 10
        #res_f *= 0
        #res_f = res_f.view(bs,nstep,-1,6)
        #res_f[:,:,:,4] = 1
        res_f = res_f.view(bs,nstep,-1)

        # delta root transoforms: G = Gd G
        delta_root = self.delta_root_mlp(steps_fr.reshape(-1,1))
        delta_root = delta_root.view(bs,nstep,-1)
        
        ## ref joints from net
        #delta_ja_ref = self.delta_joint_ref_mlp(steps_fr.reshape(-1,1))
        #delta_ja_ref = delta_ja_ref.view(bs,nstep,-1)
        
        # delta joints from net
        delta_ja_est = self.delta_joint_est_mlp(steps_fr.reshape(-1,1))
        delta_ja_est = delta_ja_est.view(bs,nstep,-1)
        delta_ja_ref = delta_ja_est
        
        vel_pred = self.vel_mlp(steps_fr.reshape(-1,1))
        vel_pred = vel_pred.view(bs,nstep,-1)
        
        return torques, delta_root, delta_ja_ref, delta_ja_est, vel_pred, res_f
    
    @staticmethod
    def rearrange_pred(est_q, est_ja, ref, est_qd, torques, res_f):
        """
        est_q:       bs,T,7
        state_qd     bs,6
        """
        bs,nstep,_ = est_q.shape

        # initial states
        state_q = torch.cat([est_q, est_ja],-1)

        # N, bs*...
        ref = ref.permute(1,0,2).reshape(nstep, -1)
        state_q = state_q.permute(1,0,2).reshape(nstep, -1)
        state_qd = est_qd.permute(1,0,2).reshape(nstep, -1)
        torques = torques.reshape(nstep, -1)
        res_f = res_f.reshape(nstep, -1,6)

        return ref, state_q, state_qd, torques, res_f
   
    def override_states(self):
        self.delta_root_mlp.override_states(self.nerf_root_rts, self.nerf_body_rts, self.bg_rts)
        self.delta_joint_est_mlp.override_states(self.nerf_body_rts)
        #self.delta_joint_ref_mlp.override_states(self.nerf_body_rts)
    
    def override_states_inv(self):
        self.delta_root_mlp.override_states_inv(self.nerf_root_rts, self.nerf_body_rts, self.bg_rts)
        self.delta_joint_est_mlp.override_states_inv(self.nerf_body_rts)

    def forward(self, frame_start=None):
        # capture requires cuda memory to be pre-allocated
        #wp.capture_begin()
        #self.graph = wp.capture_end()
        # this launch is not recorded in tape
        #wp.capture_launch(self.graph) 

        #gravity = np.interp(self.progress, [0,1], [2., 2.])
        #print('gravity mag: %f/%f'%(self.progress, gravity))
        #self.env.gravity[1] = -gravity
        if self.use_dr:
            #self.env.gravity[1] = 0
            self.env.gravity[1] = -5
        #self.env.gravity[1] = 0
        #self.env.gravity[1] = -1

        # get a batch of ref pos/orn/joints
        if frame_start is None:
            # get a batch of clips
            #frame_start = torch.rand(self.num_envs, device=self.device)
            #frame_start = (frame_start + self.opts.local_rank/self.opts.ngpu).remainder(1)
            frame_start = torch.Tensor(np.random.rand(self.num_envs)).to(self.device)
            if self.use_dr:
                #vid = np.random.randint(0,len(self.data_offset)-1)
                ##vid = 2
                #frame_start = (frame_start * (self.data_offset[vid+1]-\
                #    self.data_offset[vid]-self.wdw_length)).round()
                #frame_start = torch.clamp(frame_start, 0,np.inf).long()
                #frame_start += self.data_offset[vid]

                #frame_start = (frame_start * 30).round()
                #frame_start += self.data_offset[4] + 100

                frame_start_all = []
                #for vidid in range(len(self.data_offset)-1)[:9]:
                for vidid in self.opts.phys_vid:
                    #vidid=1
                    frame_start_sub = (frame_start * (self.data_offset[vidid+1]-\
                        self.data_offset[vidid]-self.wdw_length)).round()
                    frame_start_sub = torch.clamp(frame_start_sub, 0,np.inf).long()
                    frame_start_sub += self.data_offset[vidid]
                    frame_start_all.append(frame_start_sub)
                frame_start = torch.cat(frame_start_all,0)
                rand_list = np.asarray(range(frame_start.shape[0]))
                np.random.shuffle(rand_list)
                frame_start = frame_start[rand_list[:self.num_envs]]
                
                #frame_start = (frame_start * (self.gt_steps_visible-self.wdw_length)).round().long()
            else:
                frame_start = (frame_start * (self.gt_steps_visible-self.wdw_length)).round().long()
        else:
            frame_start = frame_start[:self.num_envs]
            ## avoid out of bound error when sample steps from the last seq
            #if frame_start == 789: pdb.set_trace()
            #max_step_limit = min(self.wdw_length, int(self.data_offset[-1]-frame_start) )
            #self.wdw_length = max_step_limit # frames
            #self.local_steps = range(self.skip_factor * self.wdw_length)
            #self.local_steps_fr = self.local_steps_fr[:self.skip_factor*max_step_limit]
            #self.wdw_length_full = len(self.local_steps)
            #self.frame2step = self.frame2step[:max_step_limit+1]

        #frame_start[:] = 0
        steps_fr = frame_start[:,None] + self.local_steps_fr[None] # bs,T
        #TODO steps that are in the same seq
        vidid,_ = fid_reindex(steps_fr[:,self.frame2step], 
                        len(self.data_offset)-1, self.data_offset)
        outseq_idx = (vidid[:,:1] - vidid)!=0

        if self.use_dr:
            with torch.no_grad():
                # get mlp data 
                batch = self.sample_sys_state(steps_fr)
                target_q, target_ja, target_qd, target_jad = batch['target_q'], \
                                                             batch['target_ja'], \
                                                             batch['target_qd'], \
                                                             batch['target_jad']
                self.target_q_vis = target_q[:,self.frame2step].clone()
                self.obj2view_vis = batch['obj2view'][:,self.frame2step].clone()
                self.ks_vis = batch['ks'][:,self.frame2step].clone()
                # cache some values
                cache_steps_fr = torch.linspace(0,self.gt_steps, self.gt_steps+1,
                                         device=self.device).long().view(1,-1)
                #batch = self.sample_sys_state(cache_steps_fr)
                #self.cache_q, self.cache_ja = batch['target_q'], batch['target_ja']
                #cache_delta_ja = self.delta_joint_est_mlp(cache_steps_fr.reshape(1,-1,1))
                #cache_delta_root = self.delta_root_mlp(cache_steps_fr.reshape(1,-1,1))
                #self.cache_ja += cache_delta_ja
                #self.cache_q = rotate_frame(self.global_q, self.cache_q) 
                #self.cache_q = self.compose_delta(self.cache_q, cache_delta_root) # delta x target
                #pdb.set_trace()
                #self.cache_ja = self.delta_joint_est_mlp(cache_steps_fr.reshape(-1,1))
                #self.cache_q = self.delta_root_mlp(cache_steps_fr.reshape(-1,1)).view(1,-1,7)
        else:
            # get mocap data
            target_q, target_ja, target_qd, target_jad = \
                                self.get_batch_input(self.amp_info_func,steps_fr, self.in_bullet)

        # transform to ground
        if not self.use_dr:
            target_q = rotate_frame(self.global_q, target_q)
            target_qd = rotate_frame_vel(self.global_q, target_qd)
        
        # compute target pos/vel
        q_at_frame = torch.cat([target_q, target_ja],-1)[:,self.frame2step]
        qd_at_frame = torch.cat([target_qd, target_jad],-1)[:,self.frame2step]
        q_at_frame = q_at_frame.permute(1,0,2).contiguous()
        qd_at_frame=qd_at_frame.permute(1,0,2).contiguous()
        target_body_q, target_body_qd, msm = ForwardKinematics.apply(q_at_frame, 
                                                               qd_at_frame, self)
        self.msm = msm
       
        beg = time.time()
        torques, delta_q, delta_ja_ref, delta_ja_est, delta_qd, res_f = self.get_net_pred(steps_fr)
        #if not self.training:
        #    res_f *= 0
        if self.use_dr:
            est_q = delta_q
            est_ja = delta_ja_est
            ref_ja = delta_ja_ref
            #delta_ja_ref = target_ja.detach() - ref_ja
            est_qd = delta_qd
        else:
            est_q = self.compose_delta(target_q, delta_q) # delta x target
            est_ja = target_ja + delta_ja_est
            ref_ja = target_ja + delta_ja_ref

            #pred_est_q_comp = compose_func(self.delta_root_mlp, self.compose_delta, target_q)
            #delta_rqd = self.compute_gradient(pred_est_q_comp, steps_fr.clone()) # grad wrt index
            ## need to add another rotation term to account delta root 
            #est_rqd = rotate_frame_vel(delta_q, target_qd)
            #est_rqd = est_rqd + delta_rqd / self.samp_int 
            #delta_jad = self.compute_gradient(self.delta_joint_est_mlp, steps_fr.reshape(-1,1).clone())
            #delta_jad = delta_jad.reshape(self.num_envs,-1,self.n_actuators)
            #est_jad = target_jad + delta_jad / self.samp_int
            #est_qd = torch.cat([ est_rqd, est_jad ], -1) + delta_qd
            est_qd = delta_qd

        ref = torch.cat([torch.zeros_like(ref_ja[...,:1].repeat(1,1,6)), ref_ja],-1)
        # compute predicted init qd: d(pred_q)/dt
        #pred_joint_qd = self.compute_gradient(self.pred_est_ja, steps_fr[:,:1]) / self.samp_int
        #pred_qd =       self.compute_gradient(self.pred_est_q_rect_rod, steps_fr[:,:1]) / self.samp_int
        #vel_pred = dvel_pred + torch.cat([ target_qd, target_jad ], -1) 
        #vel_pred = torch.cat([ vel_pred[...,:6], target_jad ], -1) 
        #vel_pred = torch.cat([ vel_pred[...,:6], pred_joint_qd ], -1) 
        #vel_pred = torch.cat([ pred_qd, pred_joint_qd ], -1)

        ref, state_q, state_qd, torques, res_f = self.rearrange_pred(
                est_q, est_ja, ref, est_qd, torques, res_f)
        # forward simulation
        res_fin = res_f.clone()
        q_init = state_q[0]
        qd_init = state_qd[0]
        if self.training:
            #TODO add some noise
            noise_ratio = np.clip(1-1.5*self.progress, 0,1)
            q_init_noise = np.random.normal(size=q_init.shape,scale=0.*noise_ratio)
            #q_init_noise = np.random.normal(size=q_init.shape,scale=0.05*noise_ratio)
            #q_init_noise = np.random.normal(size=q_init.shape,scale=0.1*noise_ratio)
            qd_init_noise = np.random.normal(size=qd_init.shape,scale=0.01*noise_ratio)
            q_init_noise = torch.Tensor(q_init_noise).to(self.device)
            qd_init_noise = torch.Tensor(qd_init_noise).to(self.device)
            # only keep the noise on root pose
            q_init_noise = q_init_noise.view(self.num_envs,-1)
            q_init_noise[:,:3] = 0
            q_init_noise[:,7:] = 0
            q_init_noise = q_init_noise.reshape(-1)

            q_init += q_init_noise
            #qd_init += qd_init_noise

        target_ke = self.target_ke[None].repeat(self.num_envs, 1).view(-1)
        target_kd = self.target_kd[None].repeat(self.num_envs, 1).view(-1)
        body_mass = self.body_mass[None].repeat(self.num_envs, 1).view(-1)
        body_qs,body_qd = ForwardWarp.apply(q_init, qd_init, torques, res_fin, 
             ref, target_ke, target_kd, body_mass, self)
       
        if self.opts.rollout:
            ## compute state pos/vel: bs, T, K,7/6, full
            state_q = state_q.reshape(self.wdw_length_full, self.num_envs, -1).permute(1,0,2)
            state_qd=state_qd.reshape(self.wdw_length_full, self.num_envs, -1).permute(1,0,2)
      
            ## use raw data
            #amp_info = self.amp_info_func(steps_fr.cpu().numpy())
            #msm = parse_amp(amp_info)
            #bullet2gl(msm,self.in_bullet)
            #state_q[...,:3]  = torch.Tensor(msm['pos']).cuda()
            #state_q[...,3:7] = torch.Tensor(msm['orn']).cuda()
            #state_q[...,7:19]= torch.Tensor(msm['jang']).cuda()
            #state_qd[...,:3]  = torch.Tensor(msm['vel']).cuda()
            #state_qd[...,3:6] = torch.Tensor(msm['avel']).cuda()
            #state_qd[...,6:18]= torch.Tensor(msm['jvel']).cuda()

            #state_q[...,1] += 0.03 # no need to account for toe since it is covered by the leg (+0.0035?)
            state_body_q, state_body_qd = fk_no_grad(state_q, state_qd, self)
        else: 
            ## compute state pos/vel: bs, T, K,7/6
            state_q = state_q[self.frame2step].reshape(self.wdw_length+1, self.num_envs, -1)
            state_qd=state_qd[self.frame2step].reshape(self.wdw_length+1, self.num_envs, -1)
            state_body_q, state_body_qd, self.tstate = ForwardKinematics.apply(state_q, 
                                                                 state_qd, self)
        
        # make sure the feet is above the ground
        if self.use_dr:
            kp_idxs =[it for it,link in enumerate(self.robot.urdf.links) if link.name in self.robot.urdf.kp_links]
            kp_idxs = [self.dict_unique_body_inv[it] for it in kp_idxs]
            foot_height = state_body_q[:,:,kp_idxs,1]
            self.state_body_kps = state_body_q[:,:,kp_idxs]
        else:
            mesh_pts,faces_single = articulate_robot_rbrt_batch(self.robot.urdf, state_body_q)
            foot_height = mesh_pts[...,1].min(-1)[0] # bs,T #TODO all foot
        #foot_offset = F.relu(-foot_height).max() # if h<0, make it 0; if h>0, don't do anything
        #target_body_q[...,1] += foot_offset # bs, 7
        #target_q[...,1] += foot_offset
        #self.global_q.data[1] += foot_offset

        if self.opts.rollout:
            # try mpc
            # plot foot
            foot_idx = [4, 8, 12, 16] #fr, fl, rr, rl
            foot_pos = state_body_q[...,foot_idx,:3]
            trimesh.Trimesh(mesh_pts[0,0].view(-1,3).cpu().detach(), faces_single).export('tmp/0.obj')
            vis_kps(foot_pos.view(-1,4,3).permute(0,2,1).cpu().detach(), 'tmp/1.obj')
            floor_scale = foot_pos.abs().max().cpu()*2
            floor_tsfm = np.eye(4)
            floor_tsfm[1,3] = 0.03
            ground_plane = trimesh.primitives.Box(
                    extents=[floor_scale, 0, floor_scale], transform=floor_tsfm)
            ground_plane.export('tmp/2.obj')

            self.controller.reset()
            msm = state_to_msm(state_q[0,0], state_qd[0,0], foot_pos[0,0], self.in_bullet)
            self.sim_robot.ResetPose(pos=msm['pos'], orn=msm['orn'])
            self.controller.update_controller_params(msm)
            self.sim_robot.ResetPose(pos=msm['pos'], orn=msm['orn'], jang=msm['jang'],
                                     vel=msm['vel'], avel=msm['avel'])
            self.msm = []
            self.obs = []
            for s in range(self.wdw_length_full):
                if s%100 == 0:
                    pdb.set_trace()
                #msm = state_to_msm(state_q[0,0], state_qd[0,0], foot_pos[0,0], self.in_bullet)
                msm = state_to_msm(state_q[s,0], state_qd[s,0], foot_pos[0,s], self.in_bullet)
                
                #self.sim_robot.SetPose(pos=msm['pos'], orn=msm['orn']) # in order to track correctly
                self.controller.update_controller_params(copy.deepcopy(msm))
                hybrid_action = self.controller.get_action()
                self.sim_robot.Step(hybrid_action)
                
                #self.sim_robot.ResetPose(pos=msm['pos'], orn=msm['orn'], jang=msm['jang'])
                # get states
                if s%self.skip_factor==0:
                    obs = self.sim_robot.GetTrueObservation()
                    self.obs.append( obs )
                    self.msm.append( msm )
            return {}

        total_loss = 0
        # root loss
        #root_target = torch.cuda.FloatTensor([1,0,0],device=self.device)
        #root_traj = body_qs[-1,0,:3]

        #root_target = torch.cuda.FloatTensor([[0,0.45,0]],device=self.device)
        #root_traj = body_qs[:, 0, :3] # S, 13, 7

        body_target = target_body_q.reshape(self.num_envs, self.wdw_length+1, -1,7)
        body_traj = body_qs.reshape(self.wdw_length+1,self.num_envs,-1,7).permute(1,0,2,3)
        body_vel =  body_qd.reshape(self.wdw_length+1,self.num_envs,-1,6).permute(1,0,2,3)

        if self.use_dr:
            #body_traj_d = body_traj[:,1:] - body_traj[:,:-1]
            #body_target_d=body_target[:,1:] - body_target[:,:-1]
            #loss_root = se3_loss(body_traj_d, body_target_d, rot_ratio=0).mean(-1)# [...,0] # bs,T, dof => bs,T
            loss_root = se3_loss(body_traj, body_target, rot_ratio=0)[...,0]# [...,0] # bs,T, dof => bs,T
            loss_body = se3_loss(body_traj, body_target, rot_ratio=0)[...,1:].mean(-1)
            loss_root = 0.2*loss_root + 0.2*loss_body
            #loss_root = loss_root + 0.1*loss_body
            #loss_root[:,0] *= 0 # set first t loss as 0
        else:
            loss_root = se3_loss(body_traj, body_target).mean(-1)
        loss_root[outseq_idx] = 0
        #loss_root[outseq_idx[:,1:]] = 0
        loss_root = clip_loss(loss_root, 0.02*self.th_multip)
        total_loss += 0.1*loss_root

        ## body loss
        #body_traj = body_qs[:, 1:] # S, 13, 7
        #body_target = body_qs[:1,1:].detach() # first frame
        #loss_pose = (body_traj - body_target).pow(2).sum()
        #total_loss += loss_pose

        ## velocity loss
        loss_vel = se3_loss(state_body_qd, target_body_qd, rot_ratio=0).mean(-1)[:,1:]
        loss_vel[outseq_idx[:,1:]] = 0
        loss_vel = clip_loss(loss_vel, 20*self.th_multip)
        #total_loss += loss_vel*1e-5

        ## vel input loss
        #loss_vel = (vel_pred[:,:1,:6] - pred_qd).norm(2,1).mean()
        #loss_vel +=(vel_pred[:,:1,6:] - pred_joint_qd).norm(2,1).mean()
        #loss_vel *= 1e-4
        #total_loss += loss_vel

        # state matching
        loss_root_state = se3_loss(state_body_q, body_traj).mean(-1)[:,1:]
        loss_root_state[outseq_idx[:,1:]] = 0
        loss_root_state = clip_loss(loss_root_state, 0.02*self.th_multip)
        total_loss += 1e-1*loss_root_state
        
        loss_vel_state = se3_loss(state_body_qd, body_vel).mean(-1)[:,1:]
        loss_vel_state[outseq_idx[:,1:]] = 0
        loss_vel_state = clip_loss(loss_vel_state, 20*self.th_multip)
        total_loss += loss_vel_state*1e-5

        ## reg
        torque_reg = torques.pow(2).mean()
        total_loss += torque_reg*1e-5
        
        res_f_reg = res_f.pow(2).mean()
        #total_loss += res_f_reg*1e-2
        total_loss += res_f_reg*5e-5
        #total_loss += res_f_reg*1e-5
        
        delta_joint_ref_reg = delta_ja_ref.pow(2).mean()
        #total_loss += delta_joint_ref_reg*1e-4

        if self.use_dr:
            foot_reg = foot_height.pow(2).view(-1)
            #foot_reg = foot_reg.mean()
            #total_loss = total_loss*0 + foot_reg*1e-2
            #print(foot_height.max())
            #print(self.global_q)
            if self.total_steps<400:
                foot_reg = foot_reg.topk(foot_reg.shape[0]*4//5, largest=False)[0].mean()
                #total_loss += foot_reg*1e-2
            else:
                foot_reg = foot_reg.topk(foot_reg.shape[0]*2//4, largest=False)[0].mean()
                #total_loss += foot_reg*1e-4
        else:
            foot_reg = foot_height.pow(2).mean()
            total_loss += foot_reg*1e-4

        #delta_joint_est_reg = delta_ja_est.pow(2).mean()
        #total_loss += delta_joint_est_reg*1e-4
        #
        #delta_q_reg = delta_q.pow(2).mean()
        #total_loss += delta_q_reg*1e-5
        #
        #delta_qd_reg = delta_qd.pow(2).mean()
        #total_loss += delta_qd_reg*5e-5

        ## smoothness
        #cache_delta_root = self.delta_root_mlp(cache_steps_fr.reshape(-1,1))[:,0]
        #cache_delta_root = se3_vec2mat(cache_delta_root)
        #root_sm_loss = compute_root_sm_2nd_loss(cache_delta_root, self.data_offset, vid=self.opts.phys_vid)
        #total_loss += 0.01*root_sm_loss

        if total_loss.isnan(): pdb.set_trace()

        # loss
        print('loss: %.4f / fw time: %.2f s'% (total_loss.cpu(), time.time() - beg))
        if len(self.total_loss_hist)>0:
            his_med = torch.stack(self.total_loss_hist,0).median()
            print(his_med)
            if total_loss > his_med * 10: 
                total_loss.zero_()
            else:
                self.total_loss_hist.append(total_loss.detach().cpu())
        else:
            self.total_loss_hist.append(total_loss.detach().cpu())
        #print(delta_joint_reg)
        #print(delta_q_reg)
        #print(loss_vel)

        loss_dict = {}
        loss_dict['total_loss'] = total_loss
        loss_dict['loss_root'] = loss_root
        loss_dict['loss_vel'] = loss_vel
        loss_dict['loss_root_state'] = loss_root_state
        loss_dict['loss_vel_state'] = loss_vel_state
        loss_dict['torque_reg'] = torque_reg
        loss_dict['res_f_reg'] = res_f_reg
        loss_dict['delta_joint_ref_reg'] = delta_joint_ref_reg
        loss_dict['foot_reg'] = foot_reg
        #print(self.target_ke)
        #print(self.target_kd)
        #print(self.body_mass)
        #loss_dict['delta_joint_est_reg'] = delta_joint_est_reg
        #loss_dict['delta_q_reg'] = delta_q_reg
        #loss_dict['delta_qd_reg'] = delta_qd_reg
        return loss_dict
    
    def backward(self, loss):
        loss.backward()

    def query(self, img_size=None):
        x_sims = []
        x_msms = []
        x_tsts = [] # target
        com_k = []
        data={}
       
        #x_rest = trimesh.Trimesh(self.gtpoints[0]*10, self.faces, 
        #        vertex_colors=self.colors, process=False)
        #in_bullet=False
        #use_urdf=False
       
        part_com = self.env.body_com.numpy()[...,None]
        part_mass = self.env.body_mass.numpy()
        x_rest = self.robot.urdf
        use_urdf=True 
        for frame in range(len(self.obs)):
            obs = self.obs[frame]
            msm = self.msm[frame]
            tst = self.tstate[frame]
            grf = self.grfs[frame]
            jaf = self.jafs[frame]

            # get com (simulated)
            com = compute_com(obs, part_com, part_mass)
            com_k.append( compute_com(msm, part_com, part_mass) )
            #x_msm = can2gym2gl(x_rest, msm, in_bullet=in_bullet, use_urdf=use_urdf, use_angle=True)
            if self.opts.rollout:
                articulate_func = bullet_can2gym2gl
            else:
                articulate_func = can2gym2gl
            x_msm = articulate_func(x_rest, msm, in_bullet=self.in_bullet, use_urdf=use_urdf)
            x_tst = articulate_func(x_rest, tst, in_bullet=self.in_bullet, use_urdf=use_urdf)
            x_sim = articulate_func(x_rest, obs, gforce=grf,com=com, in_bullet=self.in_bullet, use_urdf=use_urdf)
            #x_sim = articulate_func(x_rest, obs, gforce=grf+jaf, in_bullet=self.in_bullet, use_urdf=use_urdf)

            x_sims.append(x_sim)
            x_msms.append(x_msm)
            x_tsts.append(x_tst)
        x_sims = np.stack(x_sims,0)
        x_msms = np.stack(x_msms,0)
        x_tsts = np.stack(x_tsts,0)

        data['xs']=x_sims
        data['xgt']=x_msms
        data['tst']=x_tsts
        data['com_k']=com_k

        if img_size is not None:
            # get cameras: world to view = world to object + object to view
            # this triggers pyrender
            obj2world = self.target_q_vis[0]
            device = obj2world.device
            obj2world = se3_vec2mat(obj2world)
            world2obj = obj2world.inverse()

            obj2view = self.obj2view_vis[0]

            world2view = obj2view @ world2obj
            data['camera'] = world2view.cpu().numpy()
            data['camera'][:,3] = self.ks_vis[0].cpu().numpy()
            data['img_size'] = img_size
        return data

    def save_network(self, epoch_label):
        if self.opts.local_rank==0:
            save_dict = self.state_dict()
            param_path = '%s/params_%03d.pth'%(self.save_dir,epoch_label)
            torch.save(save_dict, param_path)

            return

    def load_network(self,model_path):
        states = torch.load(model_path,map_location='cpu')
        self.load_state_dict(states, strict=False)

def fk_no_grad(rj_q, rj_qd, self):
    """
    rj_q:  bs, T, 7+B
    rj_qd: bs, T, 6+B / none
    """
    bs,nfr,ndof = rj_q.shape

    body_q = []
    body_qd = []
    for it in range(nfr):
        rj_q_sub = rj_q[...,it,:].reshape(-1)
        rj_q_sub = wp.from_torch(rj_q_sub)
        if rj_qd is None:
            rj_qd_sub = torch.cuda.FloatTensor(np.zeros(bs*(ndof-1)))
        else:
            rj_qd_sub = rj_qd[...,it,:]
        rj_qd_sub = wp.from_torch(rj_qd_sub.reshape(-1))
        eval_fk(
            self.env,
            rj_q_sub,
            rj_qd_sub,
            None,
            self.state_steps[0]) # 
        body_q_sub = wp.to_torch(self.state_steps[0].body_q).clone() # bs*-1,7
        body_q_sub = body_q_sub.reshape(bs,-1,7)
        body_q.append( body_q_sub )
        
        body_qd_sub = wp.to_torch(self.state_steps[0].body_qd).clone() # bs*-1,6
        body_qd_sub = body_qd_sub.reshape(bs,-1,6)
        body_qd.append( body_qd_sub )
    body_q = torch.stack(body_q, 1) # bs,T,dofs,7
    body_qd = torch.stack(body_qd, 1)
    return body_q, body_qd

class ForwardKinematics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rj_q, rj_qd, self):
        """
        rj_q:  T, bs, 7+B
        rj_qd: T, bs, 6+B / none
        """
        nfr,bs,ndof = rj_q.shape
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
            
        rj_q = rj_q.view(nfr,-1)
        rj_qd=rj_qd.view(nfr,-1)
        for it,step in enumerate(self.frame2step):
            # q
            ctx.rj_q.append(wp.from_torch(rj_q[it]))
            # qd
            if rj_qd is None:
                rj_qd_sub = torch.cuda.FloatTensor(np.zeros(bs*(ndof-1)))
            else:
                rj_qd_sub = rj_qd[it]
            rj_qd_sub = wp.from_torch(rj_qd_sub)
            ctx.rj_qd.append(rj_qd_sub)

        ctx.tape = wp.Tape()
        with ctx.tape:
            body_q = []
            body_qd = []
            msm = []
            for it,step in enumerate(self.frame2step):
                step=it
                #rj_q_sub = rj_q[...,it,:].reshape(-1)
                #rj_q_sub = wp.from_torch(rj_q_sub)
                #ctx.rj_q.append(rj_q_sub)
                #if rj_qd is None:
                #    rj_qd_sub = torch.cuda.FloatTensor(np.zeros(bs*(ndof-1)))
                #else:
                #    rj_qd_sub = rj_qd[...,it,:]
                #rj_qd_sub = wp.from_torch(rj_qd_sub.reshape(-1))
                #ctx.rj_qd.append(rj_qd_sub)
                #eval_fk(
                #    self.env,
                #    rj_q_sub,
                #    rj_qd_sub,
                #    None,
                #    ctx.state_steps[step]) # 
                eval_fk(
                    self.env,
                    ctx.rj_q[it],
                    ctx.rj_qd[it],
                    None,
                    ctx.state_steps[step]) # 
                body_q_sub = wp.to_torch(ctx.state_steps[step].body_q) # bs*-1,7
                body_q_sub = body_q_sub.reshape(bs,-1,7)
                body_q.append( body_q_sub )
                msm.append(body_q_sub.detach().cpu().numpy()[0] )
                
                body_qd_sub = wp.to_torch(ctx.state_steps[step].body_qd) # bs*-1,6
                body_qd_sub = body_qd_sub.reshape(bs,-1,6)
                body_qd.append( body_qd_sub )
            body_q = torch.stack(body_q, 1) # bs,T,dofs,7
            body_qd = torch.stack(body_qd, 1)
        return body_q, body_qd, msm

    @staticmethod
    def backward(ctx, adj_body_qs, adj_body_qd, _):
        self = ctx.self
        for it,step in enumerate(self.frame2step):
            step=it
            grad_body_q = adj_body_qs[:,it].reshape(-1,7) # bs, T, -1, 7
            ctx.state_steps[step].body_q.grad = \
                    wp.from_torch(grad_body_q, dtype=wp.transform) 
            grad_body_qd = adj_body_qd[:,it].reshape(-1,6) # bs, T, -1, 7
            ctx.state_steps[step].body_qd.grad = \
                    wp.from_torch(grad_body_qd, dtype=wp.spatial_vector) 

        # return adjoint w.r.t. inputs
        ctx.tape.backward()

        rj_q_grad = [wp.to_torch(ctx.tape.gradients[i]) \
                for i in ctx.rj_q if i.requires_grad]
        if len(rj_q_grad)>0:
            rj_q_grad = torch.stack(rj_q_grad, 0).clone() # T,bs*-1
            rj_q_grad = rj_q_grad.view(-1, ctx.bs, ctx.ndof)
            rj_q_grad[rj_q_grad.isnan()] = 0
            rj_q_grad[rj_q_grad>1] = 1
        else:
            rj_q_grad = None
        
        rj_qd_grad = [wp.to_torch(ctx.tape.gradients[i]) \
                for i in ctx.rj_qd if i.requires_grad]
        if len(rj_qd_grad)>0:
            rj_qd_grad = torch.stack(rj_qd_grad, 0).clone() # T,bs*-1
            rj_qd_grad = rj_qd_grad.view(-1, ctx.bs, ctx.ndof-1)
            rj_qd_grad[rj_qd_grad.isnan()] = 0
            rj_qd_grad[rj_qd_grad>1] = 1
        else:
            rj_qd_grad = None
        
        if rj_q_grad.isnan().sum()>0:
            pdb.set_trace()
        if rj_qd_grad.isnan().sum()>0:
            pdb.set_trace()
        ctx.tape.zero()
        return (rj_q_grad, rj_qd_grad, None)

def remove_nan(q_init_grad, bs):
    """
    q_init_grad: bs*xxx
    """
    #original_shape = q_init_grad.shape
    #q_init_grad = q_init_grad.view(bs,-1)
    #invalid_batch = q_init_grad.isnan().sum(1).bool()
    #q_init_grad[invalid_batch] = 0
    #q_init_grad = q_init_grad.reshape(original_shape) 
    # clip grad
    q_init_grad[q_init_grad.isnan()] = 0
    clip_th = 0.01
    q_init_grad[q_init_grad>clip_th] = clip_th
    q_init_grad[q_init_grad<-clip_th] = -clip_th
    if q_init_grad.isnan().sum()>0:pdb.set_trace()

class ForwardWarp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_init, qd_init, torques, res_f, refs, 
            target_ke, target_kd, body_mass,self):
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
        ctx.torques = [wp.from_torch(i) for i in torques] # very slow
        ctx.res_f   = [wp.from_torch(i, dtype=wp.spatial_vector) for i in res_f] # very slow
        ctx.refs = [wp.from_torch(i) for i in refs] 
        ctx.target_ke = wp.from_torch(target_ke)
        ctx.target_kd = wp.from_torch(target_kd)
        ctx.body_mass = wp.from_torch(body_mass)

        # aux
        ctx.self = self

        # forward
        ctx.tape = wp.Tape()
        with ctx.tape:
            #TODO add kd/ke to optimization vars; not implemented by warp
            self.env.joint_target_kd = ctx.target_kd
            self.env.joint_target_ke = ctx.target_ke
            self.env.body_mass = ctx.body_mass

            # assign initial states
            eval_fk(
                self.env,
                ctx.q_init,
                ctx.qd_init,
                None,
                self.state_steps[0])
                
            # simulate
            self.grfs = []
            self.jafs = []
            for step in self.local_steps:
                self.state_steps[step].clear_forces()

                self.env.joint_target = ctx.refs[step]
                self.env.joint_act = ctx.torques[step]
                self.state_steps[step].body_f = ctx.res_f[step]

                grf, jaf = self.integrator.simulate(self.env,
                                        self.state_steps[step],
                                        self.state_steps[step+1],
                                        self.dt)
                #print(step)
                #print(self.state_steps[step].body_f.numpy().max())
                if step in self.frame2step:
                    # accumulate force to body
                    self.grfs.append(grf)
                    self.jafs.append(jaf)
    
            # get states
            obs = self.state_steps[0].body_q
            num_coords = obs.shape[0]//self.num_envs
            self.obs = [ obs.numpy()[:num_coords] ]
            wp_pos = [wp.to_torch(obs)]
            wp_vel = [wp.to_torch(self.state_steps[0].body_qd)]

            for step in self.frame2step[1:]:
                # for vis
                obs = self.state_steps[step+1].body_q
                self.obs.append( obs.numpy()[:num_coords] )
                wp_pos.append( wp.to_torch(obs) )
                wp_vel.append( wp.to_torch(self.state_steps[step+1].body_qd) )
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
        frame=1
        for step in self.frame2step[1:]:
            #print('save to step: %d/from frame: %d'%(step, frame))
            self.state_steps[step+1].body_q.grad = \
                    wp.from_torch(adj_body_qs[frame], dtype=wp.transform) 
            self.state_steps[step+1].body_qd.grad = \
                    wp.from_torch(adj_body_qd[frame], dtype=wp.spatial_vector) 
            frame+=1
        # initial step
        self.state_steps[0].body_q.grad = \
                wp.from_torch(adj_body_qs[0], dtype=wp.transform)
        self.state_steps[0].body_qd.grad = \
                wp.from_torch(adj_body_qd[0], dtype=wp.spatial_vector)

        # return adjoint w.r.t. inputs
        ctx.tape.backward()

        grad = [wp.to_torch(v) for k,v in ctx.tape.gradients.items()]
        print('max grad:')
        print(torch.cat([i.reshape(-1) for i in grad]).abs().max())

        try: 
            q_init_grad = wp.to_torch(ctx.tape.gradients[ctx.q_init]).clone()
            remove_nan(q_init_grad, self.num_envs)
        except: q_init_grad = None
        
        try: 
            qd_init_grad = wp.to_torch(ctx.tape.gradients[ctx.qd_init]).clone()
            remove_nan(qd_init_grad, self.num_envs)
        except: qd_init_grad = None
        
        refs_grad = [wp.to_torch(ctx.tape.gradients[i]) \
                for i in ctx.refs if i.requires_grad]
        if len(refs_grad)>0:
            refs_grad = torch.stack(refs_grad, 0).clone()
            remove_nan(refs_grad, self.num_envs)
        else:
            refs_grad = None
        
        torques_grad = [wp.to_torch(ctx.tape.gradients[i]) for i in ctx.torques]
        if len(torques_grad)>0:
            torques_grad = torch.stack(torques_grad, 0).clone()
            remove_nan(torques_grad, self.num_envs)
        else:
            torques_grad = None
        
        res_f_grad = [wp.to_torch(ctx.tape.gradients[i]) for i in ctx.res_f]
        if len(res_f_grad)>0:
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
        return (q_init_grad, qd_init_grad, torques_grad, res_f_grad, refs_grad, \
   target_ke_grad, target_kd_grad, body_mass_grad, None)

def can2gym2gl(x_rest, obs, gforce=None,com=None, in_bullet=False, use_urdf=False, use_angle=False):
    if use_urdf:
        if use_angle:
            cfg = np.asarray(obs['jang'])
            mesh = articulate_robot(x_rest,cfg=cfg,use_collision=True)
            rmat = R.from_quat(obs['orn']).as_matrix() # xyzw
            tmat = np.asarray(obs['pos'])
            mesh.vertices = mesh.vertices @ rmat.T + tmat[None]
        else:
            # need to parse sperical joints => assuming it's going over joints 
            mesh = articulate_robot_rbrt(x_rest, obs, gforce=gforce, com=com)
    else:
        mesh = x_rest.copy()
    return mesh

def bullet_can2gym2gl(x_rest, obs, in_bullet=False, use_urdf=False):
    rmat = R.from_quat(obs['orn']).as_matrix() # xyzw
    tmat = np.asarray(obs['pos'])
    cfg = np.asarray(obs['jang'])

    gl_to_issac = np.asarray([[0,0,1], [1,0,0], [0,1,0]])
    issac_to_gl = np.asarray([[0,1,0], [0,0,1], [1,0,0]])
    if use_urdf:
        mesh = articulate_robot(x_rest,cfg=cfg,use_collision=True)
    else:
        mesh = x_rest.copy()
    if not in_bullet:
        # transform to bullet
        mesh.vertices =  mesh.vertices @ gl_to_issac.T
    # transform in bullet
    mesh.vertices = mesh.vertices @ rmat.T + tmat[None]
    # transform back
    mesh.vertices = mesh.vertices @ issac_to_gl.T
    return mesh
