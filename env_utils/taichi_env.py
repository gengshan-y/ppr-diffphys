import torch
import pdb
import glob
import numpy as np
import taichi as ti
import trimesh
from scipy.spatial.transform import Rotation as R
real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)
mat = lambda: ti.Matrix.field(3, 3, dtype=real)


from utils.taichi_utils import MLP, zero_vec, zero_matrix, copy_outputs_slice, quat2mat, \
                    assign_xgt, clip_history
from utils.torch_utils import NeRF, clip_grad


@ti.data_oriented
class Scene:
    def __init__(self, opts, dataloader, n_grid=64, bound=3, coeff=1.5, act_strength=20, 
            dt = 5e-4, p_vol = 1, elastic_modulus = 10):
        ti.init(default_fp=ti.f32, arch=ti.gpu, flatten_if=True, device_memory_GB=9)
        #ti.init(default_fp=ti.f32, arch=ti.gpu, flatten_if=True, device_memory_GB=18)
        # hyper-params
        self.dt = dt
        self.p_vol = p_vol
        self.mu = elastic_modulus
        self.la = elastic_modulus
        self.dx = 1 / n_grid
        self.inv_dx = 1 / self.dx

        self.act_strength = act_strength
        self.n_grid = n_grid
        self.bound = bound
        self.coeff = coeff
       
        # vars
        self.actuator_id = []
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        self.n_actuators = 0

        # ti vars
        self.affine_matrix, self.deform_grad = mat(), mat()
        self.x, self.v, self.acc, self.xgt = vec(), vec(), vec(), vec()
        self.x_base = vec()
        self.x_mean = vec()
        self.jacob = vec()

        self.actuation = scalar()
        self.skin_wt = ti.field(real)
        self.gravity = scalar()
        self.banmo2world = ti.Vector.field(7, dtype=real)
        self.b2w_rmat = mat()
        self.loss_3d = scalar()

        self.grid_v_in = vec()
        self.grid_m_in = scalar()
        self.grid_v_out = vec()
    
        self.customized_robot(dataloader)
        self.add_optimizer(opts)
    
    def allocate_fields(self):
        ti.root.dense(ti.i, 3).place(self.gravity)
        ti.root.dense(ti.i, 1).place(self.banmo2world)
        ti.root.dense(ti.i, 1).place(self.b2w_rmat)
        ti.root.dense(ti.i, 1).place(self.loss_3d)
    
        ti.root.dense(ti.ij, (self.max_steps, self.n_actuators)).place(self.actuation)
        ti.root.dense(ti.k, self.max_steps).dense(ti.l, self.n_particles).place(self.x, self.v, self.affine_matrix, self.deform_grad, self.acc, self.jacob)
        ti.root.dense(ti.k, self.max_steps).place(self.x_mean)
        ti.root.dense(ti.i, self.n_particles).place(self.x_base)
        ti.root.dense(ti.k, self.gt_steps).dense(ti.l, self.n_particles).place(self.xgt)
        ti.root.dense(ti.ij, (self.n_particles, self.n_bones)).place(self.skin_wt)
        ti.root.dense(ti.ijk, self.n_grid).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)
    
        ti.root.lazy_grad()

    def customized_robot(self, dataloader):
        """
        """
        self.gtpoints = dataloader.gtpoints * 3
        self.skins = dataloader.skins
        self.faces = dataloader.faces
        self.colors= dataloader.colors
        samp_int = dataloader.samp_int

        self.n_particles = self.gtpoints[0].shape[0]
        self.gt_steps = len(self.gtpoints)
        self.max_steps = int(samp_int * self.gt_steps / self.dt)
        self.skip_factor = self.max_steps // self.gt_steps + 1
        self.n_actuators = self.skins.shape[1]*3
        self.n_bones = self.skins.shape[1]
        
        self.allocate_fields()

        # set skinning weights
        for i in range(self.n_particles):
            self.x_base[i] = self.gtpoints[0,i]
            self.deform_grad[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for j in range(self.n_bones):
                self.skin_wt[i,j] = self.skins[i,j]
        
        assign_xgt(self.xgt, self.gtpoints)
        
        self.gravity[0] = 0
        self.gravity[1] = 0 # intentially use a small garvity
        self.gravity[2] = 0
    
        # (quaternion, real first), tranlation
        self.banmo2world[0] = [1,0,0,0,  0.5,0.3,0.5]

        # add mlp
        self.mlp_act = MLP(batch_size = self.max_steps, n_actuators=self.n_actuators, 
                n_freq = 6, tscale = 1./self.max_steps) # 0-1

        #TODO
        self.mlp_act_torch = NeRF(tscale = 1./self.max_steps, N_freqs = 6, 
                        D=8, W=256,
                        out_channels=self.n_actuators,
                        in_channels_xyz=13,
                        )
        
    @ti.kernel
    def transform_x(self):
        for i in range(self.n_particles):
            self.x[0,i] = self.x_base[i]
            self.x[0,i] = self.b2w_rmat[0]@self.x[0,i]
            self.x[0,i][0] = self.x[0,i][0] + self.banmo2world[0][4]
            self.x[0,i][1] = self.x[0,i][1] + self.banmo2world[0][5]
            self.x[0,i][2] = self.x[0,i][2] + self.banmo2world[0][6]
            # ground constraints
            if self.x[0,i][1] <= 0: self.x[0,i][1]=1e-6

    def forward(self):
        ti.clear_all_gradients()
        quat2mat(self.banmo2world, self.b2w_rmat)
        self.transform_x()
        
        #affine_matrix = self.affine_matrix.to_numpy()
        #affine_det = np.abs(np.linalg.det(affine_matrix))
        #v_norm = np.linalg.norm(self.v.to_numpy(),2,-1)
        #clip_history(self.v, self.affine_matrix, v_norm, affine_det)
        
        # simulation
        input_torch = torch.Tensor(range(self.max_steps))[:,None]
        self.outputs_torch = self.mlp_act_torch.forward(input_torch) # torch ops
        self.actuation.from_torch(self.outputs_torch)
        
        for s in range(self.max_steps - 1):
            self.clear_grid()
            #self.mlp_act.forward(s, self.actuation) # original
            self.p2g(s, self.n_particles)
            self.grid_op() # gravity
            self.g2p(s, self.n_particles)

        self.loss_3d[0] = 0
        self.compute_x_avg()
        #self.compute_loss()
        return self.loss_3d
    
    def backward(self, loss):
        loss.grad[0] = 1
        #self.compute_loss.grad()
        self.compute_x_avg.grad()
        for s in reversed(range(self.max_steps - 1)):
            # Since we do not store the grid history (to save space), we redo p2g and grid op
            self.clear_grid()
            self.p2g(s, self.n_particles)
            self.grid_op()

            self.g2p.grad(s, self.n_particles)
            self.grid_op.grad()
            self.p2g.grad(s, self.n_particles)

            #self.mlp_act.grad(s, self.actuation) 
        self.outputs_torch.backward(self.actuation.grad.to_torch())
        self.transform_x.grad()
        quat2mat.grad(self.banmo2world, self.b2w_rmat)

    @ti.kernel
    def clear_grid(self):
        for i, j, k in self.grid_m_in:
            self.grid_v_in[i, j, k] = [0, 0, 0]
            self.grid_m_in[i, j, k] = 0
            self.grid_v_out[i, j, k] = [0, 0, 0]
            self.grid_v_in.grad[i, j, k] = [0, 0, 0]
            self.grid_m_in.grad[i, j, k] = 0
            self.grid_v_out.grad[i, j, k] = [0, 0, 0]
    
    @ti.kernel
    def p2g(self, f: ti.i32, n_particles: int):
        for p in range(0, n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)  # base=0 => 1/128; base=63 => 127/128
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.diag(dim=3, val=1) + self.dt * self.affine_matrix[f, p]) @ self.deform_grad[f, p]
            J = (new_F).determinant()
            self.deform_grad[f + 1, p] = new_F
    
            act_x = 0.0
            act_y = 0.0
            act_z = 0.0
            for aid in ti.static(range(self.actuation.shape[1])):
                act_x += self.actuation[f, 3*aid+0] * self.skin_wt[p, aid] * self.act_strength
                act_y += self.actuation[f, 3*aid+1] * self.skin_wt[p, aid] * self.act_strength
                act_z += self.actuation[f, 3*aid+2] * self.skin_wt[p, aid] * self.act_strength
            self.acc[f, p] = [act_x, act_y, act_z]
            self.jacob[f,p] = [J, 0, 0]
            
            A = ti.Matrix([[act_x, 0.0, 0.0], [0.0, act_y, 0.0], [0.0, 0.0, act_z]
                           ])
            cauchy = ti.Matrix(zero_matrix())
            ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            mass = 1
            
            strain = 0.5 * (new_F @ new_F.transpose() - ti.Matrix(ident))
            tr_strain = strain[0,0] + strain[1,1] + strain[2,2]
            cauchy = 2 * self.mu * strain + self.la * tr_strain * ti.Matrix(ident)

            cauchy += new_F @ A @ new_F.transpose()
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + mass * self.affine_matrix[f, p] # momentum
    
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * self.dx
                        weight = w[i](0) * w[j](1) * w[k](2)
                        self.grid_v_in[base + offset].atomic_add(
                            weight * (mass * self.v[f, p] + affine @ dpos))
                        self.grid_m_in[base + offset].atomic_add(weight * mass)
    
    
    @ti.kernel
    def g2p(self,f: ti.i32, n_particles: int):
        for p in range(0, n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector(zero_vec())
            new_C = ti.Matrix(zero_matrix())
    
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                        g_v = self.grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                        weight = w[i](0) * w[j](1) * w[k](2)
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx
    
            if new_v.norm()>1:
                new_v = ti.Vector(np.zeros(3))
                new_C = ti.Matrix(np.eye(3))
            self.v[f + 1, p] = new_v
            self.x[f + 1, p] = self.x[f, p] + self.dt * self.v[f + 1, p]
            self.affine_matrix[f + 1, p] = new_C
    
    
    @ti.kernel
    def grid_op(self):
        for i, j, k in self.grid_m_in:
            inv_m = 1 / (self.grid_m_in[i, j, k] + 1e-10)
            v_out = inv_m * self.grid_v_in[i, j, k]
            v_out[0] += self.dt*self.gravity[0]
            v_out[1] += self.dt*self.gravity[1]
            v_out[2] += self.dt*self.gravity[2]
    
            if i < self.bound and v_out[0] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
            if i > self.n_grid - self.bound and v_out[0] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
    
            if k < self.bound and v_out[2] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
            if k > self.n_grid - self.bound and v_out[2] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
    
            if j < self.bound and v_out[1] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
                normal = ti.Vector([0.0, 1.0, 0.0])
                lsq = (normal**2).sum()
                if lsq > 0.5:
                    if ti.static(self.coeff < 0):
                        v_out[0] = 0
                        v_out[1] = 0
                        v_out[2] = 0
                    else:
                        lin = (v_out.transpose() @ normal)(0)
                        if lin < 0:
                            vit = v_out - lin * normal
                            lit = vit.norm() + 1e-10
                            if lit + self.coeff * lin <= 0:
                                v_out[0] = 0
                                v_out[1] = 0
                                v_out[2] = 0
                            else:
                                v_out = (1 + self.coeff * lin / lit) * vit
            if j > self.n_grid - self.bound and v_out[1] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
    
            self.grid_v_out[i, j, k] = v_out
    
    @ti.kernel
    def compute_x_avg(self):
        for i in range(self.n_particles):
            for j in range(self.gt_steps):
                # globally transform points to the world-tilt coordinate
                x_pred = self.x[j*self.skip_factor, i]
                x_gt = self.xgt[j,i]
                # rotation
                x_gt = self.b2w_rmat[0]@x_gt
                # translation
                x_gt[0] = x_gt[0] + self.banmo2world[0][4]
                x_gt[1] = x_gt[1] + self.banmo2world[0][5]
                x_gt[2] = x_gt[2] + self.banmo2world[0][6]

                contrib = 1.0 / self.gt_steps / self.n_particles
                newdis = (x_pred - x_gt)
                dis = (newdis * newdis).sum()
                self.loss_3d[0].atomic_add(contrib * dis)

    #@ti.kernel
    #def compute_x_avg(self):
    #    for j in range(self.gt_steps):
    #        for i in range(self.n_particles):
    #            self.x_mean[j*self.skip_factor] += self.x[j*self.skip_factor, i] / self.n_particles
    #
    #@ti.kernel
    #def compute_loss(self):
    #    for j in range(self.gt_steps-1):
    #        x_mean = self.x_mean[j*self.skip_factor]
    #        x_mean_next = self.x_mean[(j+1)*self.skip_factor]
    #        dis = x_mean_next - x_mean
    #        dis_sq = (dis**2).sum()
    #        self.loss_3d[0].atomic_add(dis_sq / self.gt_steps)

    def add_optimizer(self, opts):
        params_mlp_act=[]
        for name,p in self.mlp_act_torch.named_parameters():
            params_mlp_act.append(p)
            print('optimized params: %s'%name)
        self.optimizer = torch.optim.AdamW(
        [{'params': params_mlp_act},
        ],
        lr=opts.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
                    [opts.learning_rate, # params_nerf_coarse
        ],
        opts.total_iters,
        pct_start=0.1, # use 10%
        cycle_momentum=False,
        anneal_strategy='linear',
        final_div_factor=1./5, div_factor = 25,
        )

    def update(self):
        #model.mlp_act.step(learning_rate=30)
        #for j in range(len(model.banmo2world[0])):
        #    model.banmo2world[0][j] -= 0.1*model.banmo2world.grad[0][j]
        #print(model.banmo2world)
        grad_mlp = clip_grad(self.mlp_act_torch)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def query(self):
        quat = np.asarray(self.banmo2world[0][:4]) # wxyz
        quat = [quat[1], quat[2], quat[3],quat[0]] # xyzw
        rmat = R.from_quat(quat).as_matrix()
        tmat = np.asarray(self.banmo2world[0][4:7])[None]
        np_xs = self.x.to_numpy()
        np_vs = self.v.to_numpy()
        np_as = self.acc.to_numpy()
        np_js = self.jacob.to_numpy()
        gt_meshes = []
        meshes = []
        err = []
        for frame in range(len(np_xs)):
            if frame%self.skip_factor!=0:continue
            np_x = np_xs[frame]
            np_v = np_vs[frame]
            np_a = np_as[frame]
            np_j = np_js[frame]
            print('frame: %03d, max x/v/a: %.2f, %.2f, %.2f, %.2f'\
                    %(frame, np_x.max(), np_v.max(), np_a.max(), np_j.max()))

            # apply the optimized se3
            xgt_verts = self.gtpoints[frame//self.skip_factor]
            xgt_verts = xgt_verts@rmat.T
            xgt_verts = xgt_verts + tmat 
            xgt_mesh = trimesh.Trimesh(xgt_verts, self.faces, 
                    vertex_colors=self.colors, process=False)
            x_mesh = trimesh.Trimesh(np_x, self.faces, 
                    vertex_colors=self.colors, process=False)

            # compute error
            dis = np.linalg.norm(x_mesh.vertices - xgt_mesh.vertices,2,-1)
            print('frame: %03d, mean_dis: %.2f'%(frame, 1000*dis.mean()))
            
            gt_meshes.append(xgt_mesh)
            meshes.append(x_mesh)
            err.append(dis)

        data = {'xs': meshes, 'xgt': gt_meshes, 'vs': np_vs, 'as': np_as, 'err': err}
        return data
       
