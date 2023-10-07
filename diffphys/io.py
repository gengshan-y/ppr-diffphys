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


def resize_to_nearest_multiple(image, multiple=16):
    height, width = image.shape[:2]
    new_height = int(np.ceil(height / multiple) * multiple)
    new_width = int(np.ceil(width / multiple) * multiple)
    return cv2.resize(image, (new_width, new_height))


def save_vid(
    outpath,
    frames,
    suffix=".mp4",
    upsample_frame=0,
    fps=10,
    target_size=None,
):
    """Save frames to video

    Args:
        outpath (str): Output directory
        frames: (N, H, W, x) Frames to output
        suffix (str): File type to save (".mp4" or ".gif")
        upsample_frame (int): Target number of frames
        fps (int): Target frames per second
        target_size: If provided, (H, W) target size of frames
    """
    # convert to 150 frames
    if upsample_frame < 1:
        upsample_frame = len(frames)
    frame_150 = []
    for i in range(int(upsample_frame)):
        fid = int(i / upsample_frame * len(frames))
        frame = frames[fid]
        if frame.max() <= 1:
            frame = frame * 255
        frame = frame.astype(np.uint8)
        if target_size is not None:
            frame = cv2.resize(frame, target_size[::-1])
        if suffix == ".gif":
            h, w = frame.shape[:2]
            fxy = np.sqrt(4e4 / (h * w))
            frame = cv2.resize(frame, None, fx=fxy, fy=fxy)

        # resize to make divisible by marco block size = 16
        h, w = frame.shape[:2]
        h = int(np.ceil(h / 16) * 16)
        w = int(np.ceil(w / 16) * 16)
        frame = cv2.resize(frame, (w, h))

        frame_150.append(frame)

    # to make divisible by 16
    frame_150_resized = [resize_to_nearest_multiple(frame) for frame in frame_150]
    imageio.mimsave("%s%s" % (outpath, suffix), frame_150_resized, fps=fps)
