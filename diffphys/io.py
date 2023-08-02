import cv2
import pdb
import numpy as np
import imageio
import trimesh

from diffphys.colors import label_colormap


def vis_kps(kps, path, binary_labels=None):
    """
    kps: nframe, 3+1, K
    binary_labels: nframe, k
    """
    nframe, _, nkps = kps.shape
    colormap = label_colormap()[:nkps]
    colormap = np.tile(colormap[None], (nframe, 1, 1))  # n, k, 3
    if binary_labels is not None:
        colormap = colormap * binary_labels[..., None]
    colormap = colormap.reshape((-1, 3))  # n*k, 3
    kps = np.transpose(kps[:, :3], (0, 2, 1)).reshape((-1, 3))  # n*k,3
    kps = trimesh.Trimesh(kps, vertex_colors=colormap)
    kps.export(path)


def save_vid(
    outpath,
    frames,
    suffix=".gif",
    upsample_frame=150.0,
    fps=10,
    is_flow=False,
    target_size=None,
):
    """
    save frames to video
    frames:     n,h,w,1 or n.
    """
    # convert to 150 frames
    if upsample_frame < 1:
        upsample_frame = len(frames)
    frame_150 = []
    for i in range(int(upsample_frame)):
        fid = int(i / upsample_frame * len(frames))
        frame = frames[fid]
        if is_flow:
            frame = flow_to_image(frame)
        if frame.max() <= 1:
            frame = frame * 255
        frame = frame.astype(np.uint8)
        if target_size is not None:
            frame = cv2.resize(frame, target_size[::-1])
        if suffix == ".gif":
            h, w = frame.shape[:2]
            fxy = np.sqrt(4e4 / (h * w))
            frame = cv2.resize(frame, None, fx=fxy, fy=fxy)
        frame_150.append(frame)
    try:
        imageio.mimsave("%s%s" % (outpath, suffix), frame_150, fps=fps)
    except:
        duration = len(frame_150) / fps
        imageio.mimsave("%s%s" % (outpath, suffix), frame_150, duration=duration)
