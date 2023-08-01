# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import os
import errno
from typing import Any, Dict, List, Tuple, Union
import cv2
import pdb
import configparser
import torch
import numpy as np
import imageio
import trimesh
import glob
import matplotlib.cm
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from utils.colors import label_colormap

def draw_lines(img, xy1s, xy2s):
    device = img.device
    colormap = label_colormap()
    len_colormap = colormap.shape[0]
    img = img.permute(1,2,0).cpu().numpy()*255
    img = img.astype(np.uint8)[:,:,::-1].copy()
    for i in range(len(xy1s)):
        color = tuple([int(x) for x in colormap[i%len_colormap]])
        p1 = tuple(xy1s[i].detach().cpu().numpy())
        p2 = tuple(xy2s[i].detach().cpu().numpy())
        cv2.circle(img,p1,3,  color)
        cv2.circle(img,p2,3,  color)
        cv2.line(img, p1, p2, color, thickness=1)
    #pdb.set_trace()
    #cv2.imwrite('tmp/0.png', img)
    #img = torch.Tensor(img).to(device).permute(2,0,1)[None]
    return img

def draw_pts(img, xys):
    device = img.device
    img = img.permute(1,2,0).cpu().numpy()*255
    img = img.astype(np.uint8)[:,:,::-1].copy()
    for point in xys:
        point = point.detach().cpu().numpy()
        cv2.circle(img,tuple(point),1,(0,0,255))
    #pdb.set_trace()
    #cv2.imwrite('tmp/0.png', img)
    #img = torch.Tensor(img).to(device).permute(2,0,1)[None]
    return img

def vis_kps(kps, path, binary_labels=None):
    """
    kps: nframe, 3+1, K
    binary_labels: nframe, k
    """
    nframe, _, nkps = kps.shape
    colormap = label_colormap()[:nkps]
    colormap= np.tile(colormap[None], (nframe,1,1)) # n, k, 3
    if binary_labels is not None:
        colormap= colormap * binary_labels[...,None]
    colormap= colormap.reshape((-1,3)) # n*k, 3
    kps = np.transpose(kps[:,:3],(0,2,1)).reshape((-1,3)) # n*k,3
    kps = trimesh.Trimesh(kps, vertex_colors=colormap)
    kps.export(path)

def bones_to_mesh(bones, len_max, parent=None):
    B = len(bones)
    elips_list = []
    elips = trimesh.creation.uv_sphere(radius=0.1,count=[16, 16])
    #elips = trimesh.creation.uv_sphere(radius=len_max/20,count=[16, 16])
    # remove identical vertices
    elips = trimesh.Trimesh(vertices=elips.vertices, faces=elips.faces)
    N_elips = len(elips.vertices)
    for idx,bone in enumerate(bones):
        center = bone[None,:3]
        orient = bone[3:7] # real first
        orient = orient / np.linalg.norm(orient, 2,-1)
        orient = orient[[1,2,3,0]]
        orient = R.from_quat(orient).as_matrix() # real first
        orient = orient.T # transpose R
        scale =  np.exp(bone[None, 7:10])
        if parent is not None: # joint display, use fixed size balls
            scale=0.02
        # bone coord to root coord
        elips_verts = elips.vertices
        elips_verts = elips_verts * scale
        elips_verts = elips_verts.dot(orient)
        elips_verts = elips_verts+center
        elips_sub = trimesh.Trimesh(vertices = elips_verts,faces=elips.faces)

        if parent is not None:
            if parent[idx]>-1:
                center_parent = bones[parent[idx]][None,:3]
            else:
                center_parent = center-len_max/200
            link = np.concatenate([center, center_parent],0)
            link = trimesh.creation.cylinder(len_max/100, segment=link, sections=5)
            N_link = link.vertices.shape[0]
            elips_sub = trimesh.util.concatenate([elips_sub, link])
        else:
            N_link = 0

        elips_list.append( elips_sub )
    elips = trimesh.util.concatenate(elips_list)

    colormap = label_colormap()[:B]
    colormap= np.tile(colormap[:,None], (1,N_elips+N_link,1))
    colormap[:,N_elips:] = 128
    colormap = colormap.reshape((-1,3))
    elips.visual.vertex_colors[:len(colormap),:3] = colormap
    tmp=np.sum(elips.visual.vertex_colors[:,:3]) # seems need this 
    return elips

def save_bones(bones, len_max, path, parent=None):
    elips = bones_to_mesh(bones, len_max, parent=parent)
    elips.export(path)

def vis_match(results, masks, imgs, bs,img_size,ndepth):
    # show error images
    bs = imgs.shape[0]
    for i in range(bs):
        mask_rszd = F.interpolate(masks[None],(img_size,img_size))[0,i].bool()
        img_rszd =  F.interpolate(imgs       ,(img_size,img_size))[i].permute(1,2,0)
        img_mskd = img_rszd[mask_rszd].cpu().numpy()
        
        if 'feat_err' in results.keys():
            feat_errs = results['feat_err'] 
            feat_err = feat_errs[i].view(img_size,img_size)
            feat_err[~mask_rszd] = 0.
            med = feat_err[mask_rszd].median()
            print('%d-median:%f' %(i,med))
            cv2.imwrite('tmp/match_err-%d.png'%i, (feat_err/med).cpu().numpy()*128)

    # draw lines
    if 'xyz_camera_vis' in results.keys() and 'pts_exp_vis' in results.keys():
        mask_rszd = F.interpolate(masks[None],(img_size,img_size))[0,0].bool()
        img_rszd =  F.interpolate(imgs       ,(img_size,img_size))[0].permute(1,2,0)
        xyz_coarse_frame = results['xyz_camera_vis'] 
        color_plane = torch.stack([img_rszd, torch.ones_like(img_rszd)],0).view(-1,3)
        color_plane = color_plane.cpu().numpy()
        near_plane= xyz_coarse_frame.view(bs,-1,ndepth,3)[0,:,0]
        d_near = near_plane[:,2].mean()
        near_plane[...,-1] -= d_near*0.01
        far_plane = xyz_coarse_frame.view(bs,-1,ndepth,3)[0,:,-1]
        nf_plane = torch.cat([near_plane, far_plane],0)
        #trimesh.Trimesh(nf_plane.cpu().numpy(), vertex_colors=color_plane).\
        trimesh.Trimesh(near_plane.cpu().numpy(), vertex_colors=img_rszd.view(-1,3).cpu().numpy()).\
                export('tmp/match_plane.obj')

        near_plane_mskd = near_plane[mask_rszd.view(-1)].cpu()
        pts_pred = results['pts_pred_vis'] 
        pts_pred = pts_pred[0].view(img_size,img_size,3)[mask_rszd].cpu().numpy()
        draw_lines_ray_canonical(near_plane_mskd, pts_pred,img_mskd,
                                     'tmp/match_line_pred.obj')
   
        pts_exp  = results['pts_exp_vis'] 
        pts_exp  = pts_exp[0].view(img_size,img_size,3)[mask_rszd].cpu().numpy()
        draw_lines_ray_canonical(pts_pred, pts_exp,img_mskd,
                                 'tmp/match_line_exp.obj')
    
    #pts_pred_col=results['pts_pred'][0][mask_rszd].cpu().numpy()
    #pts_exp_col = results['pts_exp'][0][mask_rszd].cpu().numpy()
    #trimesh.Trimesh(pts_pred, vertex_colors=img_mskd).export('tmp/viser_pred.obj')
    #trimesh.Trimesh(pts_exp  ,vertex_colors=img_mskd).export('tmp/viser_exp.obj')

def draw_lines_ray_canonical(near_plane_mskd, pts_exp, img_mskd, path):
    colormap = label_colormap()
    len_color = len(colormap)
    meshes = []
    idx=0
    num_pts = len(near_plane_mskd)
    for i in range(0,num_pts, num_pts//50): # display 50 points
        ## only plot idx=5
        #if idx!=5:        
        #    idx+=1
        #    continue
        segment = np.stack([near_plane_mskd[i], pts_exp[i]])
        line = trimesh.creation.cylinder(0.0001, 
                segment=segment,sections=5, vertex_colors=colormap[idx%len_color])
        meshes.append(line)
        idx+=1
    meshes = trimesh.util.concatenate(meshes)
    meshes.export(path)

def merge_dict(dict_list):
    out_dict = {}
    for k in dict_list[0].keys():
        out_dict[k] = []

    for i in range(len(dict_list)):
        for k in out_dict.keys():
            out_dict[k] += dict_list[i][k]
    return out_dict

def render_root_txt(cam_dir, cap_frame):
    # read all the data
    camlist = load_root(cam_dir, cap_frame)
    # construct camera mesh
    mesh = draw_cams(camlist)
    save_dir,seqname=cam_dir.rsplit('/',1)
    mesh.export('%s/mesh-%s.obj'%(save_dir, seqname))

def load_sils(root_dir, cap_frame):
    """
    load all the imgs with
    input is ...-(00000.png)
    """
    imglist = []
    img_path = '%s*.png'%(root_dir)
    #img_path = '%s0*.png'%(root_dir)
    all_path = sorted(glob.glob(img_path))
    if cap_frame>0:
        all_path = all_path[:cap_frame]
    for idx,path in enumerate(all_path):
        img = cv2.imread(path,0)
        imglist.append(img)
    imglist = np.asarray(imglist)
    return imglist


def load_root(root_dir, cap_frame):
    """
    load all the root se(3)
    input is ...-(00000.txt)
    """
    camlist = []
    #cam_path = '%s0*.txt'%(root_dir)
    cam_path = '%s*.txt'%(root_dir)
    all_path = sorted(glob.glob(cam_path))
    if cap_frame>0:
        all_path = all_path[:cap_frame]
    for idx,path in enumerate(all_path):
        rtk = np.loadtxt(path)
        camlist.append(rtk)
    camlist = np.asarray(camlist)
    return camlist

def draw_cams(all_cam, color='cool', axis=True,
        color_list = None):
    """
    all_cam: a list of 4x4 cameras
    """
    # scale: the scene bound
    cmap = matplotlib.cm.get_cmap(color)
    all_cam = np.asarray(all_cam)
    trans_norm = np.linalg.norm(all_cam[:,:3,3],2,-1)
    valid_cams = trans_norm>0
    trans_max = np.median(trans_norm[valid_cams])
    if np.isnan(trans_max): trans_max=0.1
    scale=trans_max
    traj_len = len(all_cam)
    cam_list = [] 
    if color_list is None:
        color_list = np.asarray(range(traj_len))/float(traj_len)
    for j in range(traj_len):
        cam_rot  = all_cam[j][:3,:3].T
        cam_tran = -cam_rot.dot(all_cam[j][:3,3:])[:,0]
    
        radius = 0.02*scale
        cam = trimesh.creation.uv_sphere(radius=radius,count=[2, 2])

        if axis:
            #TODO draw axis
            extents = np.asarray([radius*20, radius*10, radius*0.1])
            axis = trimesh.creation.axis(origin_size = radius, 
                                        origin_color = cmap(color_list[j]),
                                        axis_radius = radius* 0.1,
                                        axis_length = radius*5)
            #extents=extents)
            #axis.vertices[:,2] += radius * 5
            #cam = trimesh.util.concatenate([elips, axis])
            cam = axis

        #cam.vertices = cam.vertices + cam_tran
        cam.vertices = cam.vertices.dot(cam_rot.T) + cam_tran
        #cam.visual.vertex_colors = cmap(float(j)/traj_len)
        cam_list.append(cam)
    mesh_cam = trimesh.util.concatenate(cam_list)
    return mesh_cam

def draw_cams_pair(cam1,cam2, color='cool', axis=True,
        color_list = None):
    frame_num = cam1.shape[0]
    cam_mesh1 = draw_cams(cam1, color=color,axis=axis,color_list=color_list)
    cam_mesh2 = draw_cams(cam2, color=color,axis=axis,color_list=color_list)

    # draw line
    lines = []
    for i in range(frame_num):
        cam1_c = -cam1[i,:3,:3].T.dot(cam1[i,:3,3:])[:,0]
        cam2_c = -cam2[i,:3,:3].T.dot(cam2[i,:3,3:])[:,0]
        segment = np.stack([cam1_c, cam2_c])
        line = trimesh.creation.cylinder(0.001,segment=segment,sections=5)
        lines.append(line)
    lines = trimesh.util.concatenate(lines)
    return cam_mesh1, cam_mesh2, lines

def save_vid(outpath, frames, suffix='.gif',upsample_frame=150., fps=10,
        is_flow=False, target_size=None):
    """
    save frames to video
    frames:     n,h,w,1 or n.
    """
    # convert to 150 frames
    if upsample_frame<1: upsample_frame = len(frames)
    frame_150=[]
    for i in range(int(upsample_frame)):
        fid = int(i/upsample_frame*len(frames))
        frame = frames[fid]
        if is_flow:
            frame = flow_to_image(frame)
        if frame.max()<=1: 
            frame=frame*255
        frame = frame.astype(np.uint8)
        if target_size is not None:
            frame = cv2.resize(frame,target_size[::-1])
        if suffix=='.gif':
            h,w=frame.shape[:2]
            fxy = np.sqrt(4e4/(h*w))
            frame = cv2.resize(frame,None,fx=fxy, fy=fxy)
        frame_150.append(frame)
    imageio.mimsave('%s%s'%(outpath,suffix), frame_150, fps=fps)

class visObj(object):
    """
    a class for detectron2 vis
    """
    def has(self, name: str) -> bool:
        return name in self._fields
    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

class LineDataset(Dataset):
    '''
    '''

    def __init__(self, opts, filter_key=None, imglist=None, can_frame=0,
                    dframe=1,init_frame=0, dataid=0, numvid=1, flip=0, 
                    is_eval=False, rtk_path=None):
        super(LineDataset, self).__init__()
        self.imglist = imglist
        self.img_size = opts['img_size']
        self.num_lines = (len(imglist)-1) * self.img_size # last img not saved

        seqname = imglist[0].split('/')[-2]
        if 'load_prefix' in opts.keys():
            self.seqname = seqname
            self.prefix_seqname = opts['load_prefix']+seqname
        if rtk_path is not None:
            self.rtklist =['%s-%05d.txt'%(rtk_path, i) for i in range(len(self.imglist))]
        else:
            self.rtklist =[i.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') for i in self.imglist]

        # Load the annotation file.
        self.dataid = dataid
        print('%d lines' % self.num_lines)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        try:dataid = self.dataid
        except: dataid=0
        #TODO lolalize file
        idt = index // self.img_size# idt, idy
        idy = index %  self.img_size# idt, idy

        save_dir  = self.imglist[0].replace('JPEGImages', 'Pixels').rsplit('/',1)[0]
        if hasattr(self, 'prefix_seqname'):
            save_dir = save_dir.replace(self.seqname, self.prefix_seqname)
        
        dframe_list = [2,4,8,16,32]
        max_id = len(self.imglist)-1
        dframe_list = [1] + [i for i in dframe_list if (idt%i==0) and \
                             int(idt+i) <= max_id]
        dframe = np.random.choice(dframe_list)
    
        img_data_path = '%s/%d_%05d/img.npy'%(save_dir, dframe, idt)
        if os.path.exists(img_data_path) and 'full' in img_data_path:
            #TODO fix it
            elem={}
            data_path = '%s/%d_%05d/'%(save_dir, dframe, idt)
            elem['img']          = np.load('%s/img.npy'%(data_path),          mmap_mode="r")[...,idy,:].copy()
            elem['mask']         = np.load('%s/mask.npy'%(data_path),         mmap_mode="r")[...,idy,:].copy()
            elem['vis2d']        = np.load('%s/vis2d.npy'%(data_path),        mmap_mode="r")[...,idy,:].copy()
            elem['flow']         = np.load('%s/flow.npy'%(data_path),         mmap_mode="r")[...,idy,:].copy()
            elem['occ']          = np.load('%s/occ.npy'%(data_path),          mmap_mode="r")[...,idy,:].copy()
            elem['nml']          = np.load('%s/nml.npy'%(data_path),          mmap_mode="r")[...,idy,:].copy()
            elem['dp']           = np.load('%s/dp.npy'%(data_path),           mmap_mode="r")[...,idy,:].copy()
            #elem['dp_feat_rsmp'] = np.load('%s/dp_feat_rsmp.npy'%(data_path), mmap_mode="r")[...,idy,:].copy()
            xygrid               = np.load('%s/xygrid.npy'%(data_path),       mmap_mode="r")[...,idy,:].copy()
            xygrid = xygrid - 0.5 # xy=0.5=>first pixel, xy=111.5=>last
            dp_feat_lazy         = np.load('%s/dp_feat.npy'%(data_path),      mmap_mode="r")
            shape = elem['mask'].shape
            shape = shape[:2] + (16,) + shape[2:]
            dp_feat_rsmp = np.zeros(shape)
            for idfb in range(2):
                # vectorized
                xy_loc = xygrid[0,idfb] # 2,512
                ul_loc = np.floor(xy_loc).astype(int)  # x,y
                x = (xy_loc[0] - ul_loc[0])[:,None]
                y = (xy_loc[1] - ul_loc[1])[:,None]
                ul_loc = np.clip(ul_loc, 0,110) # clip
                q11 = dp_feat_lazy[0,idfb,:,ul_loc[1],ul_loc[0]]
                q12 = dp_feat_lazy[0,idfb,:,ul_loc[1],ul_loc[0]+1]
                q21 = dp_feat_lazy[0,idfb,:,ul_loc[1]+1,ul_loc[0]]
                q22 = dp_feat_lazy[0,idfb,:,ul_loc[1]+1,ul_loc[0]+1]
                dp_feat_rsmp_slice = (q11 * (1 - x) * (1 - y) +
                                      q21 * (1 - x) * (y - 0) +
                                      q12 * (x - 0) * (1 - y) +
                                      q22 * (x - 0) * (y - 0) )
                dp_feat_rsmp[0,idfb] = dp_feat_rsmp_slice.T

            elem['dp_feat_rsmp'] = dp_feat_rsmp[:,:,None].copy()
        elif os.path.exists(img_data_path) and 'npy' in img_data_path:
            data_path = '%s/%d_%05d/'%(save_dir, dframe, idt)
            data_path_next = '%s/%d_%05d/'%(save_dir, dframe, idt+dframe)
            elem = read_elem(data_path, idy)
            elem_next = read_elem(data_path_next, idy)
            for key in elem.keys():
                elem[key] = np.concatenate([elem[key], elem_next[key]], 1)
            elem['flow']  = np.load('%s/flow.npy'%(data_path), mmap_mode="r")[...,idy,:].copy()
            elem['occ']   = np.load('%s/occ.npy'%(data_path),  mmap_mode="r")[...,idy,:].copy()
        else:
            data_path = '%s/%d_%05d/%04d.npy'%(save_dir, dframe, idt, idy)
            elem = np.load(data_path,allow_pickle=True).item()
        # modify dataid according to training time ones

        # reload rtk based on rtk predictions
        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        # always forward flow
        idtn = idt + dframe 
        try:
            rtk_path = self.rtklist[idt]
            rtk = np.loadtxt(rtk_path)
            rtkn_path = self.rtklist[idtn]
            rtkn = np.loadtxt(rtkn_path)
            rtk = np.stack([rtk, rtkn])         
        except:
            #print('warning: loading empty camera')
            #print(rtk_path)
            rtk = np.zeros((4,4))
            rtk[:3,:3] = np.eye(3)
            rtk[:3, 3] = np.asarray([0,0,10])
            rtk[3, :]  = np.asarray([512,512,256,256]) 
            rtkn = rtk.copy()
            rtk = np.stack([rtk, rtkn])         
        
        kaug_path = '%s/%d_%05d/rtk.npy'%(save_dir, dframe, idt)
        kaug = np.load(kaug_path,allow_pickle=True).item()['kaug']
        
        #TODO fill elems
        elem['rtk']           = rtk[None]                         # 1,2,x
        elem['kaug']          = kaug                             
        elem['dataid']        = np.stack([dataid, dataid])[None] 
        elem['frameid']       = np.stack([idt,    idtn])[None]   
        elem['lineid']        = np.stack([idy,    idy])[None]   
        return elem

def read_elem(data_path, idy):
    elem={}
    elem['img']          = np.load('%s/img.npy'%(data_path),          mmap_mode="r")[...,idy,:].copy()
    elem['mask']         = np.load('%s/mask.npy'%(data_path),         mmap_mode="r")[...,idy,:].copy()
    elem['vis2d']        = np.load('%s/vis2d.npy'%(data_path),        mmap_mode="r")[...,idy,:].copy()
    elem['nml']          = np.load('%s/nml.npy'%(data_path),          mmap_mode="r")[...,idy,:].copy()
    elem['dp']           = np.zeros_like(elem['mask'])
    #elem['dp']           = np.load('%s/dp.npy'%(data_path),           mmap_mode="r")[...,idy,:].copy()
    xygrid               = np.load('%s/xygrid.npy'%(data_path),       mmap_mode="r")[...,idy,:].copy()
    xygrid = xygrid - 0.5 # xy=0.5=>first pixel, xy=111.5=>last
    dp_feat_lazy         = np.load('%s/dp_feat.npy'%(data_path),      mmap_mode="r")
    shape = elem['mask'].shape
    shape = shape[:2] + (16,) + shape[2:]
    dp_feat_rsmp = np.zeros(shape)
    # vectorized
    xy_loc = xygrid[0,0] # 2,512
    ul_loc = np.floor(xy_loc).astype(int)  # x,y
    x = (xy_loc[0] - ul_loc[0])[:,None]
    y = (xy_loc[1] - ul_loc[1])[:,None]
    ul_loc = np.clip(ul_loc, 0,110) # clip
    q11 = dp_feat_lazy[0,0,:,ul_loc[1],ul_loc[0]]
    q12 = dp_feat_lazy[0,0,:,ul_loc[1],ul_loc[0]+1]
    q21 = dp_feat_lazy[0,0,:,ul_loc[1]+1,ul_loc[0]]
    q22 = dp_feat_lazy[0,0,:,ul_loc[1]+1,ul_loc[0]+1]
    dp_feat_rsmp_slice = (q11 * (1 - x) * (1 - y) +
                          q21 * (1 - x) * (y - 0) +
                          q12 * (x - 0) * (1 - y) +
                          q22 * (x - 0) * (y - 0) )
    dp_feat_rsmp[0,0] = dp_feat_rsmp_slice.T
        
    elem['dp_feat_rsmp'] = dp_feat_rsmp[:,:,None].copy()
    return elem
    
def str_to_frame(test_frames, data_info):
    if test_frames[0]=='{':
        # render a list of videos
        idx_render = []
        for i in test_frames[1:-1].split(','):
            vid_idx = int(i)
            idx_render += range(data_info['offset'][vid_idx]-vid_idx, 
                                data_info['offset'][vid_idx+1]-vid_idx-1)
    else:
        test_frames = int(test_frames)
        if test_frames==0: 
            test_frames = data_info['len_evalloader']-1
        # render specific number of frames
        idx_render = np.linspace(0,data_info['len_evalloader']-1,
                               test_frames, dtype=int)
    return idx_render

def extract_data_info(loader):
    data_info = {}
    dataset_list = loader.dataset.datasets
    data_offset = [0]
    impath = []
    for dataset in dataset_list:
        impath += dataset.imglist
        data_offset.append(len(dataset.imglist))
    data_info['offset'] = np.asarray(data_offset).cumsum()
    data_info['impath'] = impath
    data_info['len_evalloader'] = len(loader)
    return data_info

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def reduce_component(mask):
    nb_components, output, stats, centroids = \
    cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if nb_components>1:
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
        mask = (output == max_label).astype(int)
    return mask
        
