import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # opengl seems to only work with TPU
import pyrender

import pdb
import cv2
import trimesh
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.insert(0, "%s/../../" % os.path.join(os.path.dirname(__file__)))
from diffphys.io import save_vid


class Logger:
    def __init__(self, opts):
        super(Logger, self).__init__()
        logname = "%s-%s" % (opts["seqname"], opts["logname"])
        self.save_dir = os.path.join(opts["logroot"], logname)
        self.log = SummaryWriter(self.save_dir, comment=opts["logname"])

    def show(self, tag, data, fps=10):
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
        if "vs" in data.keys():
            self.rendered_imgs["vs"] = []
        if "as" in data.keys():
            self.rendered_imgs["as"] = []
        if "err" in data.keys():
            self.rendered_imgs["err"] = []

        if "img_size" in data.keys():
            img_size = data["img_size"]
            img_size = (
                img_size[0] * img_size[2],
                img_size[1] * img_size[2],
                img_size[2],
            )
        else:
            img_size = (640, 640, 1)
        self.renderer = pyrender.OffscreenRenderer(img_size[0], img_size[1])

        # loop over data
        n_frm = len(data["sim_traj"])
        for frame in range(n_frm):
            # self.caption.text("iter:%s, frame:%04d" % (tag, frame))
            if "camera" in data.keys():
                # process the scale
                rtk = data["camera"][frame]
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
                rtk[3] = np.asarray([fl, fl, img_size[0] / 2, img_size[1] / 2])
            camera = {"rtk": rtk}

            # gt mesh
            target = data["target_traj"][frame]
            img = self.render_wdw(target, camera=camera)
            self.rendered_imgs["target"].append(img)

            # control reference
            control_ref = data["control_ref"][frame]
            img = self.render_wdw(control_ref, camera=camera)
            self.rendered_imgs["control_ref"].append(img)

            # simulated
            sim_traj = data["sim_traj"][frame]
            transparent_colors = target.visual.vertex_colors.copy()
            transparent_colors[:, 3] = 64
            target.visual.vertex_colors = transparent_colors
            merged_mesh = trimesh.util.concatenate([sim_traj, target])
            img = self.render_wdw(merged_mesh, camera=camera)
            self.rendered_imgs["sim"].append(img)

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

    def write_log(self, log_data, step):
        for k, v in log_data.items():
            self.log.add_scalar(k, v, step)

    def render_wdw(self, mesh, val=None, val_max=1, camera=None):
        if val is not None:
            if len(val.shape) == 1:
                val = np.tile(val[:, None], (1, 3))
            val = np.clip(val, -val_max, val_max)
            mesh.vertex_colors = ((val + val_max) / val_max / 2 * 255).astype(np.uint8)
        img = render_extra(self.renderer, mesh, camera["rtk"])
        return img


def render_extra(renderer, mesh, camera):
    # create scene
    plane_transform = np.eye(4)
    floor1 = trimesh.primitives.Box(extents=[20, 0, 20], transform=plane_transform)
    floor1.visual.vertex_colors[:, 0] = 10
    floor1.visual.vertex_colors[:, 1] = 200
    floor1.visual.vertex_colors[:, 2] = 60
    floor1.visual.vertex_colors[:, 3] = 128

    plane_transform[1, 3] = 0.01
    floor2 = trimesh.primitives.Box(extents=[5, 0, 5], transform=plane_transform)
    floor2.visual.vertex_colors[:, 0] = 10
    floor2.visual.vertex_colors[:, 1] = 60
    floor2.visual.vertex_colors[:, 2] = 200
    floor2.visual.vertex_colors[:, 3] = 128

    mesh = trimesh.util.concatenate([mesh, floor1, floor2])

    # mesh.export("tmp/mesh.obj")
    # pdb.set_trace()

    mesh.vertices = mesh.vertices @ camera[:3, :3].T + camera[:3, 3][None]
    scene = pyrender.Scene(ambient_light=0.4 * np.asarray([1.0, 1.0, 1.0, 1.0]))
    meshr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    meshr._primitives[0].material.RoughnessFactor = 0.5
    scene.add_node(pyrender.Node(mesh=meshr))

    cam = pyrender.IntrinsicsCamera(
        camera[3, 0], camera[3, 1], camera[3, 2], camera[3, 3], znear=1e-3, zfar=1000
    )
    cam_pose = -np.eye(4)
    cam_pose[0, 0] = 1
    cam_pose[-1, -1] = 1
    scene.add(cam, pose=cam_pose)

    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    theta = 9 * np.pi / 9
    light_pose = np.asarray(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    direc_l_node = scene.add(direc_l, pose=light_pose)

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)

    # color, _ = renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
    # cv2.imwrite("tmp/0.jpg", color)
    # pdb.set_trace()
    return color
