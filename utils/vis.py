import pdb
import numpy as np
import vedo
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, opts):
        # vedo vis
        super(Logger, self).__init__()
        dis=3
        self.camera = {'pos':[dis,0.5,dis],
                  'focalPoint': [0.5,0.5,0.5],
                  'viewup': [0,1,0]}
        self.caption = vedo.Text2D("",c="white")
        self.floor1=vedo.Plane(pos=(.5,.01,.5),normal=(0,1,0), s=(1,1), c='red8')
        self.floor2=vedo.Plane(pos=(.5,0,.5),normal=(0,1,0), s=(5,5), c='blue8')
    
        # save
        self.save_path = "%s/%s/"%(opts.checkpoint_dir, opts.logname)
        
        # tensorboard vis
        self.log = SummaryWriter(self.save_path, comment=opts.logname)


    def show(self, it, data):
        """
        xs: mesh
        xgt: mesh
        vs: values
        as: values
        err: values
        """
        # create window 
        n_wdw = 2
        resl = 320
        if 'vs' in data.keys(): n_wdw += 1
        if 'as' in data.keys(): n_wdw += 1
        if 'err' in data.keys(): n_wdw += 1
        self.plt = vedo.Plotter(shape=(1,n_wdw), size=(n_wdw*resl, resl), bg="black", 
                sharecam=True, resetcam=False)

        # loop over data
        n_frm = len(data['xs'])
        video1 = vedo.Video("%s/simu-%05d.gif"%(self.save_path, it), backend='ffmpeg', fps=10)
        video2 = vedo.Video("%s/simu-%05d.mp4"%(self.save_path, it), backend='opencv', fps=10)
        for frame in range(n_frm):
            self.wdw_idx = 0
            self.caption.text("iter:%05d, frame:%05d"%(it, frame))
          
            # prepare data
            xgt_mesh = data['xgt'][frame]
            x_mesh = data['xs'][frame]
            
            # gt mesh
            mesh = vedo.Mesh([xgt_mesh.vertices, xgt_mesh.faces])
            self.render_wdw(mesh, val=xgt_mesh.visual.vertex_colors, val_max=255)
            
            # estimated mesh
            mesh = vedo.Mesh([x_mesh.vertices, x_mesh.faces])
            self.render_wdw(mesh, val=x_mesh.visual.vertex_colors, val_max=255)
            
            # error
            if 'err' in data.keys():
                mesh = vedo.Mesh([x_mesh.vertices, x_mesh.faces])
                self.render_wdw(mesh, val=data['err'][frame], val_max=0.1)
           
            # acceleration
            if 'as' in data.keys(): 
                mesh = vedo.Mesh([x_mesh.vertices, x_mesh.faces])
                self.render_wdw(mesh, val=data['as'][frame], val_max=2)
            # velocity
            if 'vs' in data.keys(): 
                mesh = vedo.Mesh([x_mesh.vertices,x_mesh.faces])
                self.render_wdw(mesh, val=data['vs'][frame], val_max=0.5)

            video1.addFrame()
            video2.addFrame()
        video1.close()
        video2.close()
        self.plt.close()
        # TODO save to gltf (given bones etc.)

    def write_log(self, log_data, step) : 
        for k,v in log_data.items():
            self.log.add_scalar(k, v, step)

    def render_wdw(self, mesh, val=None, val_max=1):
        if val is not None:
            if len(val.shape)==1: val = np.tile(val[:,None],(1,3))
            val = np.clip(val, -val_max, val_max)
            mesh.pointdata["RGB"] = ((val+val_max)/val_max/2*255).astype(np.uint8)
            mesh.pointdata.select("RGB")
        self.plt.clear(at=self.wdw_idx)
        self.plt.show([mesh, self.floor1, self.floor2, self.caption],at=self.wdw_idx, 
                    resetcam=False, interactive=False,camera=self.camera)
        self.wdw_idx += 1
