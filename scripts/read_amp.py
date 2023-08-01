import numpy as np
import json
import sys

def parse_amp(amp_info):
  msm = {}
  msm['pos'] = amp_info[...,0:3] # root position
  msm['orn'] = amp_info[...,3:7] # root orientation (xyzw)
  msm['vel'] = amp_info[...,31:34] # root velocity
  msm['avel'] = amp_info[...,34:37] # root angular velocity
  msm['jang'] = amp_info[...,7:19] # joint angles (laikago)
  msm['jvel'] = amp_info[...,37:49] # joint angle velocity
  msm['kp'] = amp_info[...,61:73] # keypoint (4x3)
  msm['kp_vel'] = amp_info[...,73:85] # keypoint velocity
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

        feet_kp = msm['kp'] # .reshape((-1,3))
        feet_traj.append(feet_kp)

root_traj = np.stack(root_traj, 0)
feet_traj = np.stack(feet_traj, 0)
np.savetxt('%s/out-root_traj.txt'%outdir, root_traj)
np.savetxt('%s/out-feet_traj.txt'%outdir, feet_traj)
