import pdb
import math
import numpy as np
import taichi as ti   

@ti.kernel
def assign_xgt(xgt:ti.template(), gtpoints:ti.types.ndarray()):
    for i in range(xgt.shape[0]):
        for j in range(xgt.shape[1]):
            xgt[i,j][0] = gtpoints[i,j,0]
            xgt[i,j][1] = gtpoints[i,j,1]
            xgt[i,j][2] = gtpoints[i,j,2]

@ti.kernel
def clip_history(v: ti.template(), affine_matrix:ti.template(), 
                v_norm: ti.types.ndarray(), affine_det: ti.types.ndarray()):
    for s in range(v.shape[0] - 1):
        for p in range(0, v.shape[1]):
            if v_norm[s,p]>10:
                v[s,p] = ti.Vector(np.zeros(3))
            if affine_det[s,p]>1000:
                affine_matrix[s,p] = ti.Matrix(np.eye(3))

@ti.kernel
def copy_outputs(ref: ti.template(), targ: ti.template(), idx: ti.i32):
    for i in range(ref.shape[0]):
        targ[idx, i] = ref[idx, i]

@ti.kernel
def copy_outputs_slice(ref: ti.types.ndarray(), targ: ti.template(), idx: ti.i32):
    for i in range(ref.shape[0]):
        targ[idx, i] = ref[i]

@ti.data_oriented
class MLP:
    def __init__(self, batch_size, n_actuators, n_freq, tscale, n_hidden=128):
        self.tscale = tscale
        self.n_actuators = n_actuators
        self.n_freq = n_freq
        self.n_tcode = 2*n_freq+1
        self.n_hidden = n_hidden
        
        
        self.posec =  PosEmbed(batch_size, self.n_freq, self.n_tcode, self.tscale)
        self.linear_1 = Linear(batch_size, self.n_tcode, self.n_hidden)
        self.linear_2 = Linear(batch_size, self.n_hidden, self.n_actuators)
        #self.linear_2 = Linear(batch_size, self.n_hidden, self.n_hidden)
        #self.linear_3 = Linear(batch_size, self.n_hidden, self.n_actuators, with_act=False)
    
    def forward(self, tid: ti.i32,
            actuation: ti.template()):
        """
        actuation: time_steps x n_actuators
        input: weights, bias, t
        output: acutuation
        """
        self.posec.forward(tid)
        self.linear_1.forward(tid, self.posec.tcode)
        self.linear_2.forward(tid, self.linear_1.acts)
        copy_outputs(self.linear_2.acts, actuation, tid)
        #self.linear_3.forward(tid, self.linear_2.acts)
        #copy_outputs(self.linear_3.acts, actuation, tid)
    
    def grad(self, tid: ti.i32,
            actuation: ti.template()):
        """
        update params
        """
        #copy_outputs.grad(self.linear_3.acts, actuation, tid)
        #self.linear_3.forward.grad(tid, self.linear_2.acts)
        copy_outputs.grad(self.linear_2.acts, actuation, tid)
        self.linear_2.forward.grad(tid, self.linear_1.acts)
        self.linear_1.forward.grad(tid, self.posec.tcode)
        self.posec.forward.grad(tid)

    def step(self, learning_rate: ti.f32):
        """
        update params
        """
        self.linear_1.step(learning_rate)
        self.linear_2.step(learning_rate)
        #self.linear_3.step(learning_rate)

@ti.data_oriented
class Linear:
    def __init__(self, batch_size, in_features, out_features, with_act=True):
        self.in_features = in_features
        self.out_features = out_features
        self.with_act = with_act

        # declare
        weights = ti.field(dtype=ti.f32)
        bias = ti.field(dtype=ti.f32)
        acts = ti.field(dtype=ti.f32)
        ti.root.dense(ti.ij, (out_features, in_features)).place(weights)
        ti.root.dense(ti.i, out_features).place(bias)
        ti.root.dense(ti.ij, (batch_size, out_features)).place(acts)
        ti.root.lazy_grad()
   
        # init with kaiming uniform
        fan = in_features
        gain = math.sqrt(2.0)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  
        for i in range(out_features):
            for j in range(in_features):
                weights[i, j] = np.random.randn()*0.01
                #weights[i, j] = np.random.rand()*bound*2-bound

        self.weights = weights
        self.bias = bias
        self.acts = acts
    
    @ti.kernel
    def forward(self, tid: ti.i32, x: ti.template()):
        for i in range(self.out_features):
            act = 0.0
            for j in ti.static(range(self.in_features)):
                act += self.weights[i, j] * x[tid, j]
            act += self.bias[i]
            self.acts[tid, i] = ti.tanh(act)
            #if self.with_act:
            #    if act<0: act=0.
            #    else: act = ti.tanh(act)
            #self.acts[tid, i] = act
        
    def step(self, learning_rate: ti.f32):
        """
        update params
        """
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.weights[i, j] -= learning_rate * self.weights.grad[i, j]
            self.bias[i] -= learning_rate * self.bias.grad[i]
            print(self.bias.grad[i])

@ti.data_oriented
class PosEmbed:
    def __init__(self, batch_size, n_freq, n_tcode, tscale):
        self.n_freq = n_freq
        self.tscale = tscale
        self.n_tcode = n_tcode
        self.freq_bands = 2**np.linspace(0, n_freq-1, n_freq)
        
        tcode = ti.field(dtype=ti.f32)
        ti.root.dense(ti.ij, (batch_size,self.n_tcode)).place(tcode)
        ti.root.lazy_grad()

        self.tcode = tcode
    
    @ti.kernel
    def forward(self, tid: ti.i32):
        t = tid * self.tscale
        for i in ti.static(range(self.n_freq)):
            self.tcode[tid, 2*i]   = ti.sin(self.freq_bands[i]*t)
            self.tcode[tid, 2*i+1] = ti.cos(self.freq_bands[i]*t) 
        self.tcode[tid, self.n_tcode-1] = t


@ti.kernel
def copy_outputs(ref: ti.template(), targ: ti.template(), idx: ti.i32):
    for i in range(ref.shape[0]):
        targ[idx, i] = ref[idx, i]

@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]

@ti.func
def zero_matrix():
    return [zero_vec(), zero_vec(), zero_vec()]

@ti.kernel
def quat2mat(quat: ti.template(), rmat: ti.template()):
    """
    only take the first elems
    """
    r, i, j, k = quat[0][0], quat[0][1], quat[0][2], quat[0][3]
    two_s = 2.0 / (r**2+i**2+j**2+k**2)

    rmat[0][0,0] = 1 - two_s * (j * j + k * k)
    rmat[0][0,1] = two_s * (i * j - k * r)
    rmat[0][0,2] = two_s * (i * k + j * r)
    rmat[0][1,0] = two_s * (i * j + k * r)
    rmat[0][1,1] = 1 - two_s * (i * i + k * k)
    rmat[0][1,2] = two_s * (j * k - i * r)
    rmat[0][2,0] = two_s * (i * k - j * r)
    rmat[0][2,1] = two_s * (j * k + i * r)
    rmat[0][2,2] = 1 - two_s * (i * i + j * j)
