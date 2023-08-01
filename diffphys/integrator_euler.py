# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""This module contains time-integration objects for simulating
models + state forward in time.

"""

import pdb
import math
import time

import numpy as np
import warp as wp

# semi-implicit Euler integration
@wp.kernel
def integrate_bodies(body_q: wp.array(dtype=wp.transform),
                     body_qd: wp.array(dtype=wp.spatial_vector),
                     body_f: wp.array(dtype=wp.spatial_vector),
                     body_com: wp.array(dtype=wp.vec3),
                     m: wp.array(dtype=float),
                     I: wp.array(dtype=wp.mat33),
                     inv_m: wp.array(dtype=float),
                     inv_I: wp.array(dtype=wp.mat33),
                     gravity: wp.vec3,
                     dt: float,
                     body_q_new: wp.array(dtype=wp.transform),
                     body_qd_new: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    mass = m[tid]
    inv_mass = inv_m[tid]     # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, body_com[tid])
 
    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt
 
    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia*wb)   # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping, todo: expose
    w1 = w1*(1.0-0.1*dt)

    body_q_new[tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)
    body_qd_new[tid] = wp.spatial_vector(w1, v1)

@wp.kernel
def eval_body_contacts(body_q: wp.array(dtype=wp.transform),
                       body_qd: wp.array(dtype=wp.spatial_vector),
                       body_com: wp.array(dtype=wp.vec3),
                       contact_body: wp.array(dtype=int),
                       contact_point: wp.array(dtype=wp.vec3),
                       contact_dist: wp.array(dtype=float),
                       contact_mat: wp.array(dtype=int),
                       materials: wp.array(dtype=wp.vec4),
                       body_f: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    c_body = contact_body[tid]
    c_point = contact_point[tid]
    c_dist = contact_dist[tid]
    c_mat = contact_mat[tid]

    X_wb = body_q[c_body]
    v_wc = body_qd[c_body]

    # unpack spatial twist
    w = wp.spatial_top(v_wc)
    v = wp.spatial_bottom(v_wc)

    n = vec3(0.0, 1.0, 0.0)

    # transform point to world space
    cp = wp.transform_point(X_wb, c_point) - n * c_dist # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    # moment arm around center of mass
    r = cp - wp.transform_point(X_wb, body_com[c_body])

    # contact point velocity
    dpdt = v + wp.cross(w, r)     

    # check ground contact
    c = wp.dot(n, cp)

    if (c > 0.0):
        return

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    mat = materials[c_mat]

    ke = mat[0]       # restitution coefficient
    kd = mat[1]       # damping coefficient
    kf = mat[2]       # friction coefficient
    mu = mat[3]       # coulomb friction

    vn = wp.dot(n, dpdt)     
    vt = dpdt - n * vn       

    # normal force
    fn = c * ke    

    # damping force
    fd = wp.min(vn, 0.0) * kd * wp.step(c)       # again, velocity into the ground, negative

    # viscous friction
    #ft = vt*kf

    # # Coulomb friction (box)
    # lower = mu * (fn + fd)   # negative
    # upper = 0.0 - lower      # positive, workaround for no unary ops

    # vx = wp.clamp(wp.dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = wp.clamp(wp.dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    # ft = wp.vec3(vx, 0.0, vz) * wp.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), 0.0 - mu*(fn + fd))

    f_total = n * (fn + fd) + ft
    # clip force
    #warp.print(f_total[0])
    #warp.print(f_total[1])
    #warp.print(f_total[2])
    limit_val = 500. #TODO
    f_total =  wp.vec3(wp.clamp(f_total[0],-limit_val,limit_val), 
                       wp.clamp(f_total[1],-limit_val,limit_val), 
                       wp.clamp(f_total[2],-limit_val,limit_val))

    t_total = wp.cross(r, f_total)

    wp.atomic_sub(body_f, c_body, wp.spatial_vector(t_total, f_total))

# # Frank & Park definition 3.20, pg 100
@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):

    q = transform_get_rotation(t)
    p = transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    w = quat_rotate(q, w)
    v = quat_rotate(q, v) + cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):

    q = transform_get_rotation(t)
    p = transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    v = quat_rotate(q, v)
    w = quat_rotate(q, w) + cross(p, v)

    return wp.spatial_vector(w, v)


# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@wp.func
def transform_inertia(t: wp.transform, I: wp.spatial_matrix):

    t_inv = transform_inverse(t)

    q = transform_get_rotation(t_inv)
    p = transform_get_translation(t_inv)

    r1 = quat_rotate(q, vec3(1.0, 0.0, 0.0))
    r2 = quat_rotate(q, vec3(0.0, 1.0, 0.0))
    r3 = quat_rotate(q, vec3(0.0, 0.0, 1.0))

    R = mat33(r1, r2, r3)
    S = mul(skew(p), R)

    T = spatial_adjoint(R, S)
    
    return mul(mul(transpose(T), I), T)


# returns the twist around an axis
@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    
    # project imaginary part onto axis
    a = wp.vec3(q[0], q[1], q[2])
    a = wp.dot(a, axis)*axis

    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


# decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x
@wp.func
def quat_decompose(q: wp.quat):

    R = wp.mat33(
            wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)


@wp.func
def eval_joint_force(q: float,
                     qd: float,
                     target: float,
                     target_ke: float,
                     target_kd: float,
                     act: float,
                     limit_lower: float,
                     limit_upper: float,
                     limit_ke: float,
                     limit_kd: float,
                     axis: wp.vec3):

    limit_f = 0.0

    # compute limit forces, damping only active when limit is violated
    if (q < limit_lower):
        limit_f = limit_ke*(limit_lower-q) - limit_kd*min(qd, 0.0)

    if (q > limit_upper):
        limit_f = limit_ke*(limit_upper-q) - limit_kd*max(qd, 0.0)

    # joint dynamics
    total_f = (target_ke*(q - target) + target_kd*qd + act - limit_f)*axis

    return total_f


@wp.kernel
def eval_body_joints(body_q: wp.array(dtype=wp.transform),
                     body_qd: wp.array(dtype=wp.spatial_vector),
                     body_com: wp.array(dtype=wp.vec3),
                     joint_q_start: wp.array(dtype=int),
                     joint_qd_start: wp.array(dtype=int),
                     joint_type: wp.array(dtype=int),
                     joint_parent: wp.array(dtype=int),
                     joint_X_p: wp.array(dtype=wp.transform),
                     joint_X_c: wp.array(dtype=wp.transform),
                     joint_axis: wp.array(dtype=wp.vec3),
                     joint_target: wp.array(dtype=float),
                     joint_act: wp.array(dtype=float),
                     joint_target_ke: wp.array(dtype=float),
                     joint_target_kd: wp.array(dtype=float),
                     joint_limit_lower: wp.array(dtype=float),
                     joint_limit_upper: wp.array(dtype=float),
                     joint_limit_ke: wp.array(dtype=float),
                     joint_limit_kd: wp.array(dtype=float),
                     joint_attach_ke: float,
                     joint_attach_kd: float,
                     body_f: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    c_child = tid
    c_parent = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3()
    w_p = wp.vec3()
    v_p = wp.vec3()

    # parent transform and moment arm
    if (c_parent >= 0):
        X_wp = body_q[c_parent]*X_wp
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])
        
        twist_p = body_qd[c_parent]

        w_p = wp.spatial_top(twist_p)
        v_p = wp.spatial_bottom(twist_p) 
        #v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_p)

    # child transform and moment arm
    X_wc = body_q[c_child]#*X_cj
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])
    
    twist_c = body_qd[c_child]

    w_c = wp.spatial_top(twist_c)
    v_c = wp.spatial_bottom(twist_c) 
    #v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)

    # joint properties (for 1D joints)
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    type = joint_type[tid]
    axis = joint_axis[tid]

    target = joint_target[qd_start]
    target_ke = joint_target_ke[qd_start]
    target_kd = joint_target_kd[qd_start]
    limit_ke = joint_limit_ke[qd_start]
    limit_kd = joint_limit_kd[qd_start]
    limit_lower = joint_limit_lower[qd_start]
    limit_upper = joint_limit_upper[qd_start]
    
    act = joint_act[qd_start]

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # translational error
    x_err = x_c - x_p
    r_err = wp.quat_inverse(q_p)*q_c
    v_err = v_c - v_p
    w_err = w_c - w_p

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # reduce angular damping stiffness for stability
    angular_damping_scale = 0.01

    # early out for free joints
    if (type == wp.sim.JOINT_FREE):
        return

    if type == wp.sim.JOINT_FIXED:

        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2]))*wp.acos(r_err[3])*2.0

        f_total += x_err*joint_attach_ke + v_err*joint_attach_kd
        t_total += wp.transform_vector(X_wp, ang_err)*joint_attach_ke + w_err*joint_attach_kd*angular_damping_scale

    if type == wp.sim.JOINT_REVOLUTE:
        
        axis_p = wp.transform_vector(X_wp, axis)
        axis_c = wp.transform_vector(X_wc, axis)

        # swing twist decomposition
        twist = quat_twist(axis, r_err)

        q = wp.acos(twist[3])*2.0*wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
        qd = wp.dot(w_err, axis_p)

        t_total = eval_joint_force(q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p)

        # attachment dynamics
        swing_err = wp.cross(axis_p, axis_c)

        f_total += x_err*joint_attach_ke + v_err*joint_attach_kd
        t_total += swing_err*joint_attach_ke + (w_err - qd*axis_p)*joint_attach_kd*angular_damping_scale

    if type == wp.sim.JOINT_COMPOUND:

        q_off = wp.transform_get_rotation(X_cj)
        q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off
        
        # decompose to a compound rotation each axis 
        angles = quat_decompose(q_pc)

        # reconstruct rotation axes
        axis_0 = wp.vec3(1.0, 0.0, 0.0)
        q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))
        q_2 = wp.quat_from_axis_angle(axis_2, angles[2])

        q_w = q_p*q_off

        # joint dynamics
        t_total = wp.vec3()
        t_total += eval_joint_force(angles[0], wp.dot(wp.quat_rotate(q_w, axis_0), w_err), joint_target[qd_start+0], joint_target_ke[qd_start+0],joint_target_kd[qd_start+0], joint_act[qd_start+0], joint_limit_lower[qd_start+0], joint_limit_upper[qd_start+0], joint_limit_ke[qd_start+0], joint_limit_kd[qd_start+0], wp.quat_rotate(q_w, axis_0))
        t_total += eval_joint_force(angles[1], wp.dot(wp.quat_rotate(q_w, axis_1), w_err), joint_target[qd_start+1], joint_target_ke[qd_start+1],joint_target_kd[qd_start+1], joint_act[qd_start+1], joint_limit_lower[qd_start+1], joint_limit_upper[qd_start+1], joint_limit_ke[qd_start+1], joint_limit_kd[qd_start+1], wp.quat_rotate(q_w, axis_1))
        t_total += eval_joint_force(angles[2], wp.dot(wp.quat_rotate(q_w, axis_2), w_err), joint_target[qd_start+2], joint_target_ke[qd_start+2],joint_target_kd[qd_start+2], joint_act[qd_start+2], joint_limit_lower[qd_start+2], joint_limit_upper[qd_start+2], joint_limit_ke[qd_start+2], joint_limit_kd[qd_start+2], wp.quat_rotate(q_w, axis_2))
        limit_val = 10000.
        t_total =  wp.vec3(wp.clamp(t_total[0],-limit_val,limit_val), 
                           wp.clamp(t_total[1],-limit_val,limit_val), 
                           wp.clamp(t_total[2],-limit_val,limit_val))
        
        f_sub = x_err*joint_attach_ke + v_err*joint_attach_kd
        f_sub = wp.vec3(wp.clamp(f_sub[0],-limit_val,limit_val), 
                        wp.clamp(f_sub[1],-limit_val,limit_val), 
                        wp.clamp(f_sub[2],-limit_val,limit_val))
        f_total += f_sub

    # write forces
    if (c_parent >= 0):
        wp.atomic_add(body_f, c_parent, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))
        
    wp.atomic_sub(body_f, c_child, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))



@wp.func
def compute_muscle_force(
    i: int,
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    muscle_links: wp.array(dtype=int),
    muscle_points: wp.array(dtype=wp.vec3),
    muscle_activation: float,
    body_f_s: wp.array(dtype=wp.spatial_vector)):

    link_0 = muscle_links[i]
    link_1 = muscle_links[i+1]

    if (link_0 == link_1):
        return 0

    r_0 = muscle_points[i]
    r_1 = muscle_points[i+1]

    xform_0 = body_X_s[link_0]
    xform_1 = body_X_s[link_1]

    pos_0 = wp.transform_point(xform_0, r_0-body_com[link_0])
    pos_1 = wp.transform_point(xform_1, r_1-body_com[link_1])

    n = wp.normalize(pos_1 - pos_0)

    # todo: add passive elastic and viscosity terms
    f = n * muscle_activation

    wp.atomic_sub(body_f_s, link_0, wp.spatial_vector(f, wp.cross(pos_0, f)))
    wp.atomic_add(body_f_s, link_1, wp.spatial_vector(f, wp.cross(pos_1, f)))

    return 0

def compute_forces(model, state, particle_f, body_f):
    if (model.body_count and model.contact_count > 0 and model.ground):

        wp.launch(kernel=eval_body_contacts,
                  dim=model.contact_count,
                  inputs=[
                      state.body_q,
                      state.body_qd,
                      model.body_com,
                      model.contact_body0,
                      model.contact_point0,
                      model.contact_dist,
                      model.contact_material,
                      model.shape_materials
                  ],
                  outputs=[
                      body_f
                  ],
                  device=model.device)
        grf = wp.to_torch(body_f).clone()
    else:
        grf = None

    if (model.body_count):

        wp.launch(kernel=eval_body_joints,
                  dim=model.body_count,
                  inputs=[
                      state.body_q,
                      state.body_qd,
                      model.body_com,
                      model.joint_q_start,
                      model.joint_qd_start,
                      model.joint_type,
                      model.joint_parent,
                      model.joint_X_p,
                      model.joint_X_c,
                      model.joint_axis,
                      model.joint_target,
                      model.joint_act,
                      model.joint_target_ke,
                      model.joint_target_kd,
                      model.joint_limit_lower,
                      model.joint_limit_upper,
                      model.joint_limit_ke,
                      model.joint_limit_kd,
                      model.joint_attach_ke,
                      model.joint_attach_kd,
                  ],
                  outputs=[
                      body_f
                  ],
                  device=model.device)
        joint_f = wp.to_torch(body_f).clone()
        if grf is not None:
            joint_f = joint_f - grf
        else:
            grf = joint_f*0
    else:
        joint_f = None
    return grf, joint_f

class SemiImplicitIntegrator:
    """A semi-implicit integrator using symplectic Euler

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that 
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example:

        >>> integrator = wp.SemiImplicitIntegrator()
        >>>
        >>> # simulation loop
        >>> for i in range(100):
        >>>     state = integrator.forward(model, state, dt)

    """

    def __init__(self):
        pass


    def simulate(self, model, state_in, state_out, dt):

        with wp.ScopedTimer("simulate", False):

            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f
            
            if state_in.body_count:
                body_f = state_in.body_f

            grf, joint_f = compute_forces(model, state_in, particle_f, body_f)

            #-------------------------------------
            # integrate bodies

            if (model.body_count):

                wp.launch(
                    kernel=integrate_bodies,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_in.body_qd,
                        state_in.body_f,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.gravity,
                        dt,
                    ],
                    outputs=[
                        state_out.body_q,
                        state_out.body_qd
                    ],
                    device=model.device)

            return grf, joint_f
