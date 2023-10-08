import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # opengl seems to only work with TPU
import pyrender

import pdb
import cv2
import trimesh
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import sys
from diffphys.io import save_vid

sys.path.insert(0, "%s/../../" % os.path.join(os.path.dirname(__file__)))
try:
    from lab4d.utils.mesh_render_utils import PyRenderWrapper
    from lab4d.utils.vis_utils import create_floor_mesh
except:
    from diffphys.lab4d_utils import create_floor_mesh
    from diffphys.pyrender_wrapper import PyRenderWrapper


def merge_mesh(mesh_solid, mesh_transparent):
    """
    Merge two meshes and set the second mesh to be transparent
    mesh_solid: trimesh
    mesh_transparent: trimesh
    """
    transparent_colors = mesh_transparent.visual.vertex_colors.copy()
    transparent_colors[:] = 64
    mesh_transparent.visual.vertex_colors = transparent_colors
    merged_mesh = trimesh.util.concatenate([mesh_solid, mesh_transparent])
    return merged_mesh


class PhysVisualizer:
    def __init__(self, save_dir):
        super(PhysVisualizer, self).__init__()
        self.save_dir = save_dir
        self.log = SummaryWriter(self.save_dir)
        self.floor = create_floor_mesh()

    def show(self, tag, data, fps=10, view_mode="ref"):
        """
        sim_traj: mesh
        target_traj: mesh
        vs: values
        as: values
        err: values
        """
        # convert tag
        if isinstance(tag, int):
            tag = "%05d" % tag

        # create window
        self.rendered_imgs = {"target": [], "sim": [], "control_ref": []}
        if "distilled_traj" in data.keys():
            self.rendered_imgs["distilled"] = []
        if "vs" in data.keys():
            self.rendered_imgs["vs"] = []
        if "as" in data.keys():
            self.rendered_imgs["as"] = []
        if "err" in data.keys():
            self.rendered_imgs["err"] = []

        if "img_size" in data.keys():
            img_size = data["img_size"]
            img_size = (
                int(img_size[0] * img_size[2]),
                int(img_size[1] * img_size[2]),
                img_size[2],
            )
        else:
            img_size = (640, 640, 1)
        self.renderer = PyRenderWrapper(img_size[:2])  # h,w
        self.renderer.set_light_topdown(gl=True)
        self.view_mode = view_mode

        # DEBUG
        if "distilled_traj" in data.keys():
            self.visualize_trajectory(data["distilled_traj"], "distilled_traj-" + tag)
        self.visualize_trajectory(data["sim_traj"], "sim_traj-" + tag)

        # loop over data
        n_frm = len(data["sim_traj"])
        for frame in range(n_frm):
            # self.caption.text("iter:%s, frame:%04d" % (tag, frame))
            if "camera" in data.keys():
                # process the scale
                rtk = data["camera"][frame].copy()
                rtk[3] *= img_size[2]
            else:
                rtk = np.eye(4)
                # rotate the camera by 45 degrees along x axis
                rtk[:3, :3] = rtk[:3, :3] @ np.asarray(
                    cv2.Rodrigues(np.asarray([-5 * np.pi / 6, 0, 0]))[0]
                )
                rtk[:3, :3] = rtk[:3, :3] @ np.asarray(
                    cv2.Rodrigues(np.asarray([0, -np.pi / 2, 0]))[0]
                )
                rtk[:3, 3] = np.asarray([0.0, 0.0, 3.0])
                fl = max(img_size[0], img_size[1])
                rtk[3] = np.asarray([fl, fl, img_size[1] / 2, img_size[0] / 2])
            camera = {"rtk": rtk}

            # gt mesh
            target = data["target_traj"][frame]
            img = self.render_wdw(target, camera=camera)
            self.rendered_imgs["target"].append(img)

            # control reference
            control_ref = data["control_ref"][frame]
            merged_mesh = merge_mesh(control_ref, target)
            img = self.render_wdw(merged_mesh, camera=camera)
            self.rendered_imgs["control_ref"].append(img)

            # simulated
            sim_traj = data["sim_traj"][frame]
            merged_mesh = merge_mesh(sim_traj, target)
            img = self.render_wdw(merged_mesh, camera=camera)
            self.rendered_imgs["sim"].append(img)

            # distilled
            if "distilled_traj" in data.keys():
                distilled = data["distilled_traj"][frame]
                merged_mesh = merge_mesh(distilled, target)
                img = self.render_wdw(merged_mesh, camera=camera)
                self.rendered_imgs["distilled"].append(img)

            # error
            if "err" in data.keys():
                img = self.render_wdw(
                    data["sim_traj"][frame],
                    val=data["err"][frame],
                    val_max=0.1,
                    camera=camera,
                )
                self.rendered_imgs["err"].append(img)

            # acceleration
            if "as" in data.keys():
                img = self.render_wdw(
                    data["sim_traj"][frame],
                    val=data["as"][frame],
                    val_max=2,
                    camera=camera,
                )
                self.rendered_imgs["as"].append(img)
            # velocity
            if "vs" in data.keys():
                img = self.render_wdw(
                    data["sim_traj"][frame],
                    val=data["vs"][frame],
                    val_max=0.5,
                    camera=camera,
                )
                self.rendered_imgs["vs"].append(img)
        self.renderer.delete()

        # combine all
        all_imgs = []
        for i in range(n_frm):
            img = np.concatenate(
                [frames[i] for frames in self.rendered_imgs.values()], axis=1
            )
            all_imgs.append(img)
        self.rendered_imgs["all"] = all_imgs

        for key, frames in self.rendered_imgs.items():
            save_vid(
                "%s/%s-%s" % (self.save_dir, key, tag),
                frames,
                suffix=".mp4",
                upsample_frame=0,
                fps=fps,
            )
        # TODO save to gltf (given bones etc.)

    def visualize_trajectory(self, trajs, tag):
        trajs = trajs.copy()
        skip_num = len(trajs) // 10  # keep 10 frames
        trajs = trajs[::skip_num]
        max_w = 2  # max width
        floor = self.floor.copy()
        floor.vertices *= len(trajs) / floor.vertices[:, 0].max() / 2 * 1.2 * max_w

        meshes = [floor]
        for idx, mesh in enumerate(trajs):
            mesh = mesh.copy()
            # center mesh
            mesh.vertices[:, 0] -= mesh.vertices[:, 0].mean()
            mesh.vertices[:, 0] += max_w * (idx - (len(trajs) - 1) / 2)
            meshes.append(mesh)
        meshes = trimesh.util.concatenate(meshes)

        meshes.export("%s/%s.obj" % (self.save_dir, tag))

    def write_log(self, log_data, step):
        for k, v in log_data.items():
            self.log.add_scalar(k, v, step)

    def render_wdw(self, mesh, val=None, val_max=1, camera=None):
        if val is not None:
            if len(val.shape) == 1:
                val = np.tile(val[:, None], (1, 3))
            val = np.clip(val, -val_max, val_max)
            mesh.vertex_colors = ((val + val_max) / val_max / 2 * 255).astype(np.uint8)
        img = render_image(
            self.renderer, mesh, self.floor, camera["rtk"], view=self.view_mode
        )
        return img


def render_image(renderer, mesh, scene, camera, view="ref"):
    """
    Render a mesh with a scene

    Args:
        renderer: PyRenderWrapper
        mesh: trimesh
        scene: trimesh
        camera: (4,4) rt (3x4) k (1x4), fx,fy,px,py
    """
    input_dict = {}
    mesh = trimesh.util.concatenate([mesh, scene])
    input_dict["shape"] = mesh
    if view == "ref":
        # ref view camera
        scene_to_cam = np.eye(4)
        scene_to_cam[:3] = camera[:3]
        renderer.set_camera(scene_to_cam)
    elif view == "bev":
        # bev camera
        renderer.set_camera_bev(6, gl=True)
    elif view == "front":
        # frontal view
        renderer.set_camera_frontal(6, delta=0.0, gl=True)
    renderer.set_intrinsics(camera[3])
    color = renderer.render(input_dict)[0]
    return color
