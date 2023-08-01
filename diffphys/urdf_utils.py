import pdb
import time
import cv2
import numpy as np
import trimesh
import torch
from diffphys.geom_utils import vec_to_sim3, se3exp_to_vec, se3_vec2mat, se3_mat2rt,\
                                fid_reindex
from scipy.spatial.transform import Rotation as R

def robot2parent_idx(urdf):
    """
    get parent idx from urdf
    """
    ball_joint=urdf.ball_joint
    name2joints_idx = urdf.name2joints_idx
    parent_idx = [-1] + [0] * len(name2joints_idx.keys())
    for idx,link in enumerate(urdf._reverse_topo):
        path = urdf._paths_to_base[link]
        # potentially connected to root
        if len(path) == 2:
            joint = urdf._G.get_edge_data(path[0], path[1])['joint']
            if joint.name in name2joints_idx.keys():
                joint_idx = name2joints_idx[joint.name]
                parent_idx[joint_idx+1] = 0
            
        if len(path)>2:
            for jdx in range(len(path)-1):
                # find the current joint
                joint = urdf._G.get_edge_data(path[jdx], path[jdx+1])['joint']
                if joint.name in name2joints_idx.keys():
                    joint_idx = name2joints_idx[joint.name]
                    for kdx in range(jdx+1, len(path)-1):
                        # find the next joint
                        next_joint = urdf._G.get_edge_data(path[kdx], path[kdx+1])['joint']
                        if next_joint.name in name2joints_idx.keys():
                            next_joint_idx = name2joints_idx[next_joint.name]
                            parent_idx[joint_idx+1] = next_joint_idx+1
                            break
                    break
    
    #link_map = {}
    #for idx,link in enumerate(urdf._reverse_topo):
    #    link_map[link.name] = idx
    #parent_idx = []
    #for idx,link in enumerate(urdf._reverse_topo):
    #    path = urdf._paths_to_base[link]
        #if len(path)>1:
        #    if ball_joint and idx%3!=1:continue
        #    parent = path[1]
        #    if ball_joint:
        #        parent_idx.append( (link_map[parent.name]+2)//3 )
        #    else:
        #        parent_idx.append( link_map[parent.name] )
        #else:
        #    parent_idx.append(-1)
    return parent_idx

def get_joints(urdf,device="cpu"):
    """
    return joint locations wrt parent link
    a root joint of (0,0,0) is added
    joints: physical joints, B
    name2joints_idx, name to joint idx
    angle_names, registered angle predictions
    only called in diffphys/robot.py
    """
    ball_joint=urdf.ball_joint
    counter = 0
    name2joints_idx = {}
    name2query_idx = {}
    joints = []
    angle_names = []
    for idx,joint in enumerate(urdf.joints):
        if joint.joint_type == 'fixed':continue
        angle_names.append(joint.name)
        if ball_joint and idx%3!=2: continue
        name2query_idx[joint.name] = counter
        counter += 1
    counter = 0
    for idx,joint in enumerate(urdf.joints):
        if joint.joint_type == 'fixed':continue
        if ball_joint and idx%3!=0: continue
        # for ball joints, only the first has non zero center
        name2joints_idx[joint.name] = counter
        origin = torch.Tensor(joint.origin[:3,3]).to(device)
        joints.append(origin)
        counter += 1
    
    joints = torch.stack(joints,0)
    urdf.name2joints_idx = name2joints_idx # contain all physics joints
    urdf.name2query_idx = name2query_idx # contain all physics joints
    urdf.angle_names = angle_names # contain all dofs
    return joints 


def vis_joints(robot):
    # deprecated: origin is the transform from the parent link to the child link.
    pts = []
    for joint in robot.joints:
        pts.append(joint.origin[:3,3])
    pts = np.stack(pts,0)
    trimesh.Trimesh(pts).export('tmp/0.obj')

# angles to config
def angles2cfg(robot, angles):
    cfg = {}
    for idx,name in enumerate(robot.angle_names):
        cfg[name] = angles[idx].cpu().numpy()
    return cfg

def get_visual_origin(urdf):
    lfk = urdf.link_fk()

    rt = {}
    for link in lfk:
        for visual in link.visuals:
            if len(visual.geometry.meshes)>0:
                for mesh in visual.geometry.meshes:
                    rt[mesh] = visual.origin
    return rt

def get_collision_origin(urdf):
    lfk = urdf.link_fk()

    rt = {}
    for link in lfk:
        for visual in link.collisions:
            if len(visual.geometry.meshes)>0:
                for mesh in visual.geometry.meshes:
                    rt[mesh] = visual.origin
    return rt

def articulate_robot_rbrt_batch(robot, rbrt):
    """
    Note: this assumes rbrt is a torch tensor
    robot: urdfpy object
    rbrt: ...,13,7, first is the root pose instead of the base link
    returns a mesh of the articulated robot
    """
    ndim = rbrt.ndim
    device = rbrt.device
    fk = get_collision_origin(robot)

    # store a mesh
    verts_all=[]
    faces_single=[]
    face_base = 0
    count = 0
    for it,item in enumerate(fk.items()):
        if it not in robot.unique_body_idx:
            continue
        tm,pose = item
        pose = np.reshape(pose, (1,)*(ndim-2)+(4,4))
        pose = torch.Tensor(pose).to(device)

        pose = se3_vec2mat(rbrt[...,count,:]) @ pose
        #pose = se3_vec2mat(rbrt[...,it,:]) @ pose
        rmat, tmat = se3_mat2rt(pose) # ...,3,3
    
        verts = np.reshape(tm.vertices, (1,)*(ndim-2)+(-1,3,))
        verts = torch.Tensor(verts).to(device) # ...,3
        permute_tup = tuple(range(ndim-2))
        verts = verts @ torch.permute(rmat, permute_tup+(-1,-2)) + tmat[...,None,:]
        verts_all.append(verts)
        # add faces of each part        
        faces = tm.faces
        faces += face_base
        face_base += verts.shape[-2] 
        faces_single.append(faces)
        count += 1
    verts_all = torch.cat(verts_all, -2)
    faces_single = np.concatenate(faces_single, -2)
    return verts_all, faces_single

def articulate_robot_rbrt(robot, rbrt, gforce=None, com=None):
    """
    robot: urdfpy object
    rbrt: 13,7
    returns a mesh of the articulated robot
    """
    fk = get_collision_origin(robot)
    #fk = get_visual_origin(robot)

    # store a mesh
    meshes=[]
    count = 0
    for it,item in enumerate(fk.items()):
        if it not in robot.unique_body_idx:
            continue
        tm,pose = item
        pose = se3_vec2mat(rbrt[count]) @ pose
        #pose = se3_vec2mat(rbrt[it]) @ pose
        rmat, tmat = se3_mat2rt(pose)

        faces = tm.faces
        faces -= faces.min()
        tm = trimesh.Trimesh(tm.vertices, tm.faces) # ugly workaround for uv = []
        tm=tm.copy()
        tm.vertices = tm.vertices.dot(rmat.T) + tmat[None]
    
        # add arrow mesh
        if gforce is not None:
            force = gforce[count, 3:6].cpu().numpy()
            mag = np.linalg.norm(force, 2,-1)
            if mag>10: # a threshold
            #if mag>0:
                orn = force/mag
                orth1 = np.cross(orn, [0,0,1])
                orth2 = np.cross(orn, orth1)
                transform = np.eye(4)
                transform[:3,3] = tm.vertices.mean(0)
                transform[:3,2] = orn
                transform[:3,1] = orth1 / np.linalg.norm(orth1)
                transform[:3,0] = -orth2 / np.linalg.norm(orth2)

                arrow = get_arrow(mag, transform)

                tm = trimesh.util.concatenate([tm, arrow])
                tm.visual.vertex_colors[:,0] = 255
                tm.visual.vertex_colors[:,1] = 0
                tm.visual.vertex_colors[:,2] = 0
            else:
                tm.visual.vertex_colors[:,0] = 255
                tm.visual.vertex_colors[:,1] = 255
                tm.visual.vertex_colors[:,2] = 255

        meshes.append(tm)
        count += 1
    vertex_colors = np.concatenate([i.visual.vertex_colors for i in meshes],0)
    meshes = trimesh.util.concatenate(meshes)
    meshes.visual.vertex_colors = vertex_colors

    # com
    if com is not None:
        transform = np.eye(4)
        transform[:3,3] = com
        transform[:3,2] = [0,-1,0]
        transform[:3,1] = [1,0,0]
        transform[:3,0] = [0,0,-1]
        arrow = get_arrow(60, transform)
        arrow.visual.vertex_colors[:,0] = 0# np.clip(255*mag, 128, 255)
        arrow.visual.vertex_colors[:,1] = 255
        arrow.visual.vertex_colors[:,2] = 0
        meshes = trimesh.util.concatenate([meshes, arrow])

    return meshes

def get_arrow(mag, transform):
    mag = np.clip( mag/200, 0,1)
    box = trimesh.primitives.Box(extents=[0.05,0.05,1*mag])
    con = trimesh.creation.cone(0.05, 0.1)
    con.vertices[:,2] += 0.5 * mag
    arrow = trimesh.util.concatenate([box, con])
    arrow.vertices[:,2] += 0.5 * mag # 0,0,1 direction

    arrow.vertices = arrow.vertices @ transform[:3,:3].T + transform[:3,3][None]
    return arrow

def articulate_robot(robot, cfg=None, use_collision=False):
    """
    robot: urdfpy object
    returns a mesh of the articulated robot in its original scale
    """
    if cfg is not None and type(cfg) is not dict:
        cfg = torch.Tensor(cfg)
        cfg = angles2cfg(robot, cfg)
    if use_collision:
        fk = robot.collision_trimesh_fk(cfg=cfg)
    else:
        fk = robot.visual_trimesh_fk(cfg=cfg)

    # store a mesh
    meshes=[]
    for tm in fk:
        pose = fk[tm]
        faces = tm.faces
        faces -= faces.min()
        tm = trimesh.Trimesh(tm.vertices, tm.faces) # ugly workaround for uv = []
        tm=tm.copy()
        tm.vertices = tm.vertices.dot(pose[:3,:3].T) + pose[:3,3][None]
        meshes.append(tm)
    meshes = trimesh.util.concatenate(meshes)
    return meshes

def render_robot(robot, save_path, cfg=None, use_collision=False):
    """Visualize the URDF in a given configuration.
    modified from urdfpy
    Parameters
    ----------
    cfg : dict
        A map from joints or joint names to configuration values for
        each joint. If not specified, all joints are assumed to be
        in their default configurations.
    use_collision : bool
        If True, the collision geometry is visualized instead of
        the visual geometry.
    """
    import os
    os.environ["PYOPENGL_PLATFORM"] = "egl" # for offscreen rendering
    import pyrender  # Save pyrender import for here for CI

    meshes = articulate_robot(robot, cfg=cfg, use_collision=use_collision)

    # add mesh
    scene = pyrender.Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
    scene.add(pyrender.Mesh.from_trimesh(meshes, smooth=False))
    #pyrender.Viewer(scene, use_raymond_lighting=True)
    # add camera and lighting
    cam = pyrender.OrthographicCamera(xmag=.5, ymag=.5)
    cam_pose = np.eye(4);
    cam_pose[:3,:3] = cv2.Rodrigues(np.asarray([0.,-np.pi/4*2,0.]))[0]
    cam_pose[0,3]=-1
    cam_node = scene.add(cam, pose=cam_pose)
    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
    light_pose = np.eye(4);
    light_pose[:3,:3] = cv2.Rodrigues(np.asarray([-np.pi/8*3.,-np.pi/8*5,0.]))[0]
    light_pose[0,3]=-1
    direc_l_node = scene.add(direc_l, pose=light_pose)

    r = pyrender.OffscreenRenderer(256,256)
    color,_ = r.render(scene,\
      flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | \
            pyrender.RenderFlags.SKIP_CULL_FACES)
    r.delete()
    cv2.imwrite(save_path, color)
    return color, meshes
