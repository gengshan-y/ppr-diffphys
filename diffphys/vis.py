import os
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
import pyrender

import pdb
import trimesh
import numpy as np
import vedo
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '%s/../../'%os.path.join(os.path.dirname(__file__)))
from diffphys.io import save_vid

class Logger:
    def __init__(self, opts):
        # vedo vis
        super(Logger, self).__init__()
        dis=5
        self.camera = {'pos':[dis,dis/2,dis],
                  'focalPoint': [0.5,0.5,0.5],
                  'viewup': [0,1,0]}
        self.caption = vedo.Text2D("",c="black")
        self.floor1=vedo.Plane(pos=(.5,.01,.5),normal=(0,1,0), s=(dis/2,dis/2), c='blue8', alpha=0.8)
        self.floor2=vedo.Plane(pos=(.5,0,.5),normal=(0,1,0), s=(dis*2,dis*2), c='green8',alpha=0.8)
    
        # save
        self.save_path = "%s/%s/"%(opts.checkpoint_dir, opts.logname)
        
        # tensorboard vis
        self.log = SummaryWriter(self.save_path, comment=opts.logname)


    def show(self, tag, data):
        """
        xs: mesh
        xgt: mesh
        vs: values
        as: values
        err: values
        """
        # convert tag
        if isinstance(tag, int):
            tag='%05d'%tag

        # create window 
        n_wdw = 2
        resl = 320
        if 'vs' in data.keys(): n_wdw += 1
        if 'as' in data.keys(): n_wdw += 1
        if 'err' in data.keys(): n_wdw += 1
        if 'tst' in data.keys(): n_wdw += 1
        
        if 'camera' in data.keys(): use_gui = False
        else: use_gui = True

        if use_gui:
            self.plt = vedo.Plotter(shape=(1,n_wdw), size=(n_wdw*resl, resl), bg="white", 
                sharecam=True, resetcam=False)
            video1 = vedo.Video("%s/simu-%s.gif"%(self.save_path, tag), backend='ffmpeg', fps=10)
            video2 = vedo.Video("%s/simu-%s.mp4"%(self.save_path, tag), backend='opencv', fps=10)
            # find the center x/z location
            vis_offset = np.stack([i.vertices for i in data['xgt']],0)[0].mean(0)
            vis_offset[1] = 0; vis_offset = vis_offset[None]
        else:
            vis_offset = np.asarray([[0,0,0]])
            self.rendered_imgs = [[] for i in range(n_wdw)]
            img_size = data['img_size']
            img_size = (img_size[0] * img_size[2], img_size[1] * img_size[2], img_size[2])
            self.renderer = pyrender.OffscreenRenderer(img_size[0], img_size[1])

        # loop over data
        n_frm = len(data['xs'])
        for frame in range(n_frm):
            self.wdw_idx = 0
            self.caption.text("iter:%s, frame:%04d"%(tag, frame))
            if use_gui:
                camera = None
            else:
                # process the scale
                rtk = data['camera'][frame]
                rtk[3] *= img_size[2]
                camera = {'rtk': rtk}
          
            # prepare data
            xgt_mesh = data['xgt'][frame]
            x_mesh = data['xs'][frame]
            
            # gt mesh
            mesh = vedo.Mesh([xgt_mesh.vertices - vis_offset, xgt_mesh.faces])
            self.render_wdw(mesh, val=xgt_mesh.visual.vertex_colors, val_max=255, camera=camera)
           
            if 'tst' in data.keys():
                tst_mesh = data['tst'][frame]
                mesh = vedo.Mesh([tst_mesh.vertices - vis_offset, tst_mesh.faces])
                self.render_wdw(mesh, val=tst_mesh.visual.vertex_colors, val_max=255, camera=camera)
            
            # estimated mesh
            mesh = vedo.Mesh([x_mesh.vertices - vis_offset, x_mesh.faces])
            self.render_wdw(mesh, val=x_mesh.visual.vertex_colors, val_max=255, camera=camera)

            # error
            if 'err' in data.keys():
                mesh = vedo.Mesh([x_mesh.vertices - vis_offset, x_mesh.faces])
                self.render_wdw(mesh, val=data['err'][frame], val_max=0.1, camera=camera)
           
            # acceleration
            if 'as' in data.keys(): 
                mesh = vedo.Mesh([x_mesh.vertices - vis_offset, x_mesh.faces])
                self.render_wdw(mesh, val=data['as'][frame], val_max=2, camera=camera)
            # velocity
            if 'vs' in data.keys(): 
                mesh = vedo.Mesh([x_mesh.vertices - vis_offset,x_mesh.faces])
                self.render_wdw(mesh, val=data['vs'][frame], val_max=0.5, camera=camera)

            if use_gui:
                video1.addFrame()
                video2.addFrame()

        if use_gui:
            video1.close()
            video2.close()
            self.plt.close()
        else:
            for i in range(n_wdw):
                save_vid("%s/vid%d-simu-%s"%(self.save_path, i, tag), 
                        self.rendered_imgs[i],suffix='.gif',upsample_frame=0)
                save_vid("%s/vid%d-simu-%s"%(self.save_path, i, tag), 
                        self.rendered_imgs[i],suffix='.mp4',upsample_frame=0)
            self.renderer.delete()
        # TODO save to gltf (given bones etc.)

    def write_log(self, log_data, step) : 
        for k,v in log_data.items():
            self.log.add_scalar(k, v, step)

    def render_wdw(self, mesh, val=None, val_max=1, camera=None):
        if val is not None:
            if len(val.shape)==1: val = np.tile(val[:,None],(1,3))
            val = np.clip(val, -val_max, val_max)
            mesh.pointdata["RGB"] = ((val+val_max)/val_max/2*255).astype(np.uint8)
            mesh.pointdata.select("RGB")
        if camera is None: 
            camera = self.camera
            self.plt.clear(at=self.wdw_idx)
            self.plt.show([mesh, self.floor1, self.floor2, self.caption],at=self.wdw_idx, 
                        resetcam=False, interactive=False,camera=camera)
            self.wdw_idx += 1
        else:
            img = render_extra(self.renderer, mesh, camera['rtk'])
            self.rendered_imgs[self.wdw_idx].append( img )
            self.wdw_idx += 1


def render_extra(renderer, mesh, camera):
    mesh = trimesh.Trimesh(mesh.points(), mesh.faces(), vertex_colors=mesh.pointdata["RGB"])
    plane_transform = np.eye(4); plane_transform[1,1] = -1
    floor = trimesh.primitives.Box(extents=[10, 0, 10], transform=plane_transform)
    floor.visual.vertex_colors[:,:3] = 192
    mesh = trimesh.util.concatenate([mesh, floor])

    mesh.vertices = mesh.vertices @ camera[:3,:3].T + camera[:3,3][None]
    scene = pyrender.Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
    meshr = pyrender.Mesh.from_trimesh(mesh,smooth=False)
    meshr._primitives[0].material.RoughnessFactor=.5
    scene.add_node( pyrender.Node(mesh=meshr ))
    
    cam = pyrender.IntrinsicsCamera(
            camera[3,0],
            camera[3,1],
            camera[3,2],
            camera[3,3],
            znear=1e-3,zfar=1000)
    cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
    scene.add(cam, pose=cam_pose)

    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    theta = 9*np.pi/9
    light_pose = np.asarray([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
    direc_l_node = scene.add(direc_l, pose=light_pose)

    color, _ = renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    #color, _ = renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
    #cv2.imwrite('tmp/0.jpg', color)
    return color
