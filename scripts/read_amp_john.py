import os
import numpy as np
import json
import sys
import trimesh
import pdb
sys.path.insert(0, '%s/../../'%os.path.join(os.path.dirname(__file__)))
from utils.io import vis_kps

def parse_amp(amp_info):
  msm = {}
  msm['pos'] = amp_info[...,0:3] # root position
  msm['orn'] = amp_info[...,3:6] # root orientation (xyzw)
  msm['kp'] = amp_info[...,6:18] # keypoint (4x3)
  return msm


path = sys.argv[1]
outdir = sys.argv[2]

root_traj = []
feet_traj = []
with open(path, "r") as f:
    amp_info = json.load(f)
    samp_int = amp_info['FrameDuration']
    amp_info = np.asarray(amp_info['Frames'])
    for i in range(len(amp_info)):
        msm = parse_amp(amp_info[i])
        root_pose = np.concatenate((msm['pos'], msm['orn']), 0)
        root_traj.append(root_pose)

        feet_kp = msm['kp'].reshape((-1,3)).T
        #feet_kp = msm['kp'].reshape((3,-1))
        feet_traj.append(feet_kp)

root_traj = np.stack(root_traj, 0)
feet_traj = np.stack(feet_traj, 0)

vis_kps(feet_traj, 'tmp/1.obj')
