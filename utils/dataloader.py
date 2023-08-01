import os
import glob
import pdb
import numpy as np
import trimesh
import json

class DataLoader:
    def __init__(self, opts, cap=-1):
        super(DataLoader).__init__()
        datadir = './data/motion_sequences/%s/'%opts.seqname
        # load mesh
        gtmeshes = [trimesh.load(i,process=False) \
                for i in sorted(glob.glob('./%s/*-mesh-*.obj'%(datadir)))]
        cams = [np.loadtxt(i) \
                for i in sorted(glob.glob('./%s/*-cam-*.txt'%(datadir))) if 'bg' not in i]
        bgcams = [np.loadtxt(i) \
                for i in sorted(glob.glob('./%s/*-cam-*.txt'%(datadir))) if 'bg' in i]
        if cap>0: 
            gtmeshes = gtmeshes[:cap]
            cams = cams[:cap]
            bgcams = bgcams[:cap]

        if len(gtmeshes)>0:
            self.gtpoints = np.asarray([x.vertices for x in gtmeshes])
            self.faces = gtmeshes[0].faces
            self.colors= gtmeshes[0].visual.vertex_colors
            self.skins = np.load('./%s/val-rest-skin.npy'%(datadir))
        self.samp_int = 0.1 # s
        #self.samp_int = 0.02 # s
        if os.path.exists('%s/amp-%s.txt'%(datadir, opts.seqname)):
            with open ('%s/amp-%s.txt'%(datadir, opts.seqname), "r") as f:
                self.amp_info = json.load(f)
                self.samp_int = self.amp_info['FrameDuration']
                self.amp_info = np.asarray(self.amp_info['Frames'])

        self.data_info = {'offset': np.asarray([0,len(self.amp_info)])}

def parse_amp(amp_info):
  msm = {}
  msm['pos'] = amp_info[...,0:3]
  msm['orn'] = amp_info[...,3:7]
  msm['vel'] = amp_info[...,31:34]
  msm['avel'] = amp_info[...,34:37]
  msm['jang'] = amp_info[...,7:19]
  msm['jvel'] = amp_info[...,37:49]
  msm['kp'] = amp_info[...,61:73]
  msm['kp_vel'] = amp_info[...,73:85]
  return msm

