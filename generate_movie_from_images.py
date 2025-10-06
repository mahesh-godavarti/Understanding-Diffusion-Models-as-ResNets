#!/usr/bin/env python3
import os, sys, glob
import numpy as np

def list_frames(input_dir, prefix="step_", ext=".png"):
    paths = sorted(glob.glob(os.path.join(input_dir, f"{prefix}*{ext}")))
    if not paths:
        raise FileNotFoundError(f"No frames found in {input_dir} matching {prefix}*{ext}")
    return paths

def read_frames(paths):
    import imageio.v2 as iio
    frames = [iio.imread(p) for p in paths]
    # Ensure uint8
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    return frames

def write_mp4_imageio(frames, out_path, fps=5):
    import imageio.v2 as iio
    # Try to use ffmpeg plugin
    with iio.get_writer(out_path, format="FFMPEG", mode="I", fps=fps, codec="libx264") as w:
        for f in frames:
            # Ensure 3 channels
            if f.ndim == 2:
                f = np.stack([f]*3, axis=-1)
            elif f.shape[-1] == 4:
                f = f[..., :3]
            w.append_data(f)

def write_mp4_opencv(frames, out_path, fps=5):
    import cv2
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open. Install proper codecs.")
    for f in frames:
        if f.ndim == 2:
            f = np.stack([f]*3, axis=-1)
        if f.shape[-1] == 4:
            f = f[..., :3]
        # imageio reads RGB; OpenCV expects BGR
        f_bgr = f[..., ::-1]
        vw.write(f_bgr)
    vw.release()

def write_gif(frames, out_path, fps=5):
    import imageio.v2 as iio
    duration = 1.0 / float(fps)
    iio.mimsave(out_path, frames, duration=0.1*duration)

def make_movie(input_dir="./grids_per_step", out_mp4="progress.mp4", out_gif="progress.gif", fps=5):
    paths = list_frames(input_dir)
    frames = read_frames(paths)

    # Try MP4 via imageio+ffmpeg
    try:
        import imageio_ffmpeg  # ensures ffmpeg binary is available
        _ = imageio_ffmpeg.get_ffmpeg_exe()
        write_mp4_imageio(frames, out_mp4, fps=fps)
        print(f"[done] wrote MP4 via imageio/ffmpeg: {out_mp4}")
        return
    except Exception as e:
        print(f"[warn] imageio/ffmpeg path failed: {e}")

    # Try MP4 via OpenCV
    try:
        import cv2  # noqa
        write_mp4_opencv(frames, out_mp4, fps=fps)
        print(f"[done] wrote MP4 via OpenCV: {out_mp4}")
        return
    except Exception as e:
        print(f"[warn] OpenCV path failed: {e}")

    # Fallback to GIF
    try:
        write_gif(frames, out_gif, fps=fps)
        print(f"[done] wrote GIF fallback: {out_gif}")
        return
    except Exception as e:
        print(f"[error] GIF fallback failed: {e}")
        raise

if __name__ == "__main__":
    # customize if you like
    make_movie("./grids_per_step", "progress.mp4", "progress.gif", fps=10)

