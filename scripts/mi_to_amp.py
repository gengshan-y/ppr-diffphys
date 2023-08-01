import os
import sys
import pdb
import numpy as np
import json
import trimesh
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'')
from nnutils.urdf_utils import articulate_robot

seqname=sys.argv[1]
data_path = '%s/amp-%s-bk.txt'%(seqname, seqname)
output_path = '%s/amp-%s.txt'%(seqname, seqname)
with open (data_path, "r") as f:
    amp_info = json.load(f)
    frames_info = amp_info['Frames']

robot_to_can = np.asarray([[0,1,0,0], [0,0,1,0], [1,0,0,0], [0,0,0,1]])
robot = URDF.load('utils/tds/data/laikago/laikago_toes_zup_joint_order.urdf')
links = ['toeFR', 'toeFL', 'toeRR', 'toeRL']

for idx in range(len(frames_info)):
    frame_info = frames_info[idx]
    pos = frame_info[0:3]
    orn = frame_info[3:7]
    ang = frame_info[7:19]

    # rotate frame
    T = np.eye(4)
    T[:3,:3] = R.from_quat(orn).as_matrix()
    T[:3,3] = pos
    T = T @ robot_to_can
    frame_info[0:3] = T[:3,3]
    frame_info[3:7] = R.from_matrix(T[:3,:3]).as_quat()
    frame_info += [0]*42

    # add kp attribute 
    cfg = {}
    jdx = 0
    for joint in robot.joints:
        joint_name = joint.name
        if 'jtoe' in joint_name: continue
        cfg[joint_name] = ang[jdx]
        jdx += 1
    kp_dict = robot.link_fk(cfg=cfg) # body frame
    kp = []
    for kp_link, kp_T in kp_dict.items():
        if kp_link.name in links:
            kp.append(kp_T[:,3])
    kp = np.stack(kp,0).T
    kp = T@kp # 3+1,4 (world frame)
    frame_info += kp[:3].T.flatten().tolist()
    frame_info += [0] * 12
    frames_info[idx] = frame_info

    # visualize points
    if idx==0:
        robot.angle_names = list(cfg.keys())
        mesh = articulate_robot(robot,cfg=list(cfg.values()),use_collision=True)
        mesh.vertices =mesh.vertices@ T[:3,:3].T + T[:3,3][None]
        mesh.export('tmp/robot.obj')

# compute velocities
frames_np = np.asarray(frames_info)
pos =  frames_np[:,0:3]
rot =  frames_np[:,3:7]
joint =frames_np[:,7:19]
kp =frames_np[:,61:73]

pos_vel = np.zeros_like(pos)
pos_vel[1:-1] = pos[2:] - pos[:-2]
pos_vel[0] = pos_vel[1]
pos_vel[-1] = pos_vel[-2]
pos_vel /= (2*amp_info['FrameDuration'])

joint_vel = np.zeros_like(joint)
joint_vel[1:-1] = joint[2:] - joint[:-2]
joint_vel[0] = joint_vel[1]
joint_vel[-1] = joint_vel[-2]
joint_vel /= (2*amp_info['FrameDuration'])

rot_vel = R.from_quat(rot).as_rotvec() # xyz
rot_vel[1:-1] = rot_vel[2:] - rot_vel[:-2]
rot_vel[0] = rot_vel[1]
rot_vel[-1] = rot_vel[-2]
rot_vel /= (2*amp_info['FrameDuration'])

kp_vel = np.zeros_like(kp)
kp_vel[1:-1] = kp[2:] - kp[:-2]
kp_vel[0] = kp_vel[1]
kp_vel[-1] = kp_vel[-2]
kp_vel /= (2*amp_info['FrameDuration'])

frames_np[:,31:34] = pos_vel
frames_np[:,34:37] = rot_vel
frames_np[:,37:49] = joint_vel
frames_np[:,73:85] = kp_vel
frames_info = frames_np.tolist()

# visualize final trajectories
trns = [i[0:3] for i in frames_info]
trns = np.stack(trns,0)
trimesh.Trimesh(trns).export('tmp/trns.obj')

kps = [i[61:73] for i in frames_info]
kps = np.stack(kps,0)
#kps = np.reshape(kps,(-1,4,3))[:,0]
kps = np.reshape(kps,(-1,3))
trimesh.Trimesh(kps).export('tmp/kps.obj')


amp_info['Frames'] = frames_info
with open(output_path, 'w') as fp:
    json.dump(amp_info, fp, indent=4)
