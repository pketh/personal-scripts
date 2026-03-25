#!/usr/bin/env python3
"""
Chroma Key — Remove a solid background color from an MP4 and export:
  • WebM  (VP8 + alpha)      — Chrome, Firefox, Edge
  • MOV   (HEVC + alpha)     — Safari on macOS (uses hevc_videotoolbox, macOS only)
  • MOV   (ProRes 4444)      — Safari fallback on all platforms (lossless alpha)

HEVC alpha encoding notes:
  - macOS:          hevc_videotoolbox gives true compressed HEVC+alpha (small file)
  - Linux/Windows:  libx265 silently drops the alpha channel — ProRes 4444 is used
                    instead, which Safari has supported since macOS 10.6 / iOS 9

GPU acceleration is used automatically when available:
  • Frame keying:   OpenCV CUDA (NVIDIA) → OpenCL/Metal → CPU fallback
  • FFmpeg encoding: hevc_videotoolbox (Apple) → hevc_nvenc (NVIDIA) → libx265

Usage:
    python chroma_key.py input.mp4 "#00FF00"
    python chroma_key.py input.mp4 "#00FF00" --tolerance 30 --softness 3
    python chroma_key.py input.mp4 "#FFFFFF" --webm-only
    python chroma_key.py input.mp4 "#FFFFFF" --hevc-only
    python chroma_key.py input.mp4 "#00FF00" --no-gpu

Requirements:
    pip install opencv-python numpy          # CPU only
    pip install opencv-contrib-python numpy  # + CUDA keying (NVIDIA)
    FFmpeg must be on PATH (brew install ffmpeg  /  https://ffmpeg.org)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# GPU capability detection
# ---------------------------------------------------------------------------

class GPUInfo:
    def __init__(self, force_cpu: bool = False):
        self.force_cpu        = force_cpu
        self.cuda_available   = False
        self.opencl_available = False
        self.cuda_device_name = ""

        if force_cpu:
            return

        # NVIDIA CUDA via OpenCV
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                self.cuda_available = True
                cv2.cuda.setDevice(0)
                self.cuda_device_name = cv2.cuda.DeviceInfo().name()
        except (cv2.error, AttributeError):
            pass

        # OpenCL fallback (Apple Metal, Intel, AMD)
        if not self.cuda_available:
            try:
                cv2.ocl.setUseOpenCL(True)
                self.opencl_available = cv2.ocl.useOpenCL() and cv2.ocl.haveOpenCL()
            except Exception:
                pass

    @property
    def use_cuda(self) -> bool:
        return self.cuda_available and not self.force_cpu

    @property
    def use_opencl(self) -> bool:
        return self.opencl_available and not self.cuda_available and not self.force_cpu

    def summary(self) -> str:
        if self.force_cpu:     return "CPU (forced)"
        if self.use_cuda:      return f"CUDA  ({self.cuda_device_name})"
        if self.use_opencl:    return "OpenCL  (Metal / integrated GPU)"
        return "CPU"


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: #{h}. Must be 6 hex digits.")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def hex_to_hsv_range(
    hex_color: str,
    tolerance: int = 30,
    value_tolerance: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    bgr = np.uint8([[list(hex_to_bgr(hex_color))]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    lower = np.array([max(0,   h - tolerance), max(0, s - 80),  max(0,   v - value_tolerance)])
    upper = np.array([min(179, h + tolerance), 255,             min(255, v + value_tolerance)])
    return lower, upper


# ---------------------------------------------------------------------------
# Per-frame alpha mask  (CPU / CUDA / OpenCL paths)
# ---------------------------------------------------------------------------

def build_alpha_cpu(frame_bgr, lower, upper, softness):
    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    alpha = cv2.bitwise_not(cv2.inRange(hsv, lower, upper))
    if softness > 0:
        k = softness * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return alpha


def build_alpha_opencl(frame_bgr, lower, upper, softness):
    """UMat path — zero-copy on Apple Metal / integrated GPUs."""
    u = cv2.UMat(frame_bgr)
    hsv   = cv2.cvtColor(u, cv2.COLOR_BGR2HSV)
    alpha = cv2.bitwise_not(cv2.inRange(hsv, lower, upper))
    if softness > 0:
        k = softness * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return cv2.UMat.get(alpha)


def build_alpha_cuda(frame_bgr, lower, upper, softness, _cache):
    """NVIDIA CUDA path — reuses pre-allocated GpuMat objects."""
    if "gpu_bgr" not in _cache:
        h, w = frame_bgr.shape[:2]
        _cache["gpu_bgr"]   = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        _cache["gpu_hsv"]   = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        _cache["gpu_alpha"] = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)

    _cache["gpu_bgr"].upload(frame_bgr)
    cv2.cuda.cvtColor(_cache["gpu_bgr"], cv2.COLOR_BGR2HSV, dst=_cache["gpu_hsv"])
    gpu_mask = cv2.cuda.inRange(_cache["gpu_hsv"], lower.tolist(), upper.tolist())
    cv2.cuda.bitwise_not(gpu_mask, dst=_cache["gpu_alpha"])

    if softness > 0:
        k = softness * 2 + 1
        if "gauss" not in _cache or _cache.get("gauss_k") != k:
            _cache["gauss"]   = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (k, k), 0)
            _cache["gauss_k"] = k
        _cache["gauss"].apply(_cache["gpu_alpha"], dst=_cache["gpu_alpha"])

    return _cache["gpu_alpha"].download()


def make_keyer(gpu: GPUInfo):
    """Return a keying function with signature (frame_bgr, lower, upper, softness) -> alpha."""
    cuda_cache = {}
    if gpu.use_cuda:
        return lambda f, lo, hi, s: build_alpha_cuda(f, lo, hi, s, cuda_cache)
    if gpu.use_opencl:
        return build_alpha_opencl
    return build_alpha_cpu


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(input_path, hex_color, tolerance, softness, tmp_dir, gpu):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Error: Cannot open '{input_path}'")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lower, upper = hex_to_hsv_range(hex_color, tolerance=tolerance)
    key_frame    = make_keyer(gpu)

    print(f"\n[1/3] Keying frames  ({width}x{height} @ {fps:.2f}fps, ~{total} frames)  [{gpu.summary()}]")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        alpha = key_frame(frame, lower, upper, softness)
        bgra  = cv2.merge([*cv2.split(frame), alpha])
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{idx:06d}.png"), bgra)
        idx += 1
        if idx % 30 == 0:
            pct = idx / total * 100 if total > 0 else 0
            print(f"  {idx}/{total}  ({pct:.0f}%)", end="\r")

    cap.release()
    print(f"  {idx} frames extracted.          ")
    return fps, width, height, idx


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def encode_webm(tmp_dir, output_path, fps, gpu):
    """VP8 + alpha — best encoder available."""
    encoder, extra = "libvpx", []
    if not _force_cpu and _has_encoder("vp8_vaapi"):
        encoder = "vp8_vaapi"
        extra   = ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=yuva420p,hwupload"]

    print(f"\n[2/3] Encoding WebM  (VP8+alpha, {encoder}) -> {output_path}")
    _ffmpeg([
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v", encoder,
        "-auto-alt-ref", "0",
        "-b:v", "0", "-crf", "10",
        "-pix_fmt", "yuva420p",
        *extra,
        output_path,
    ])
    print(f"  Saved: {output_path}")


def encode_safari_mov(tmp_dir, output_path, fps, gpu):
    """
    HEVC+alpha on macOS (hevc_videotoolbox) — compressed, small file.
    ProRes 4444 everywhere else — lossless alpha, larger file but universally supported by Safari.
    """
    frame_pat = os.path.join(tmp_dir, "frame_%06d.png")
    is_mac    = platform.system() == "Darwin"

    # --- macOS: hardware HEVC+alpha ---
    if is_mac and _has_encoder("hevc_videotoolbox"):
        print(f"\n[3/3] Encoding HEVC+alpha  (hevc_videotoolbox) -> {output_path}")
        ok = _ffmpeg([
            "-framerate", str(fps),
            "-i", frame_pat,
            "-c:v", "hevc_videotoolbox",
            "-allow_sw", "1",
            "-alpha_quality", "1",
            "-tag:v", "hvc1",
            output_path,
        ], check=False)
        if ok:
            print(f"  Saved: {output_path}  [HEVC+alpha, hvc1]")
            return

    # --- fallback: ProRes 4444 (lossless alpha, Safari-native) ---
    print(f"\n[3/3] Encoding ProRes 4444 (lossless alpha, Safari-native) -> {output_path}")
    if not is_mac:
        print("  (libx265 silently drops alpha on Linux/Windows; ProRes 4444 is used instead)")
    _ffmpeg([
        "-framerate", str(fps),
        "-i", frame_pat,
        "-c:v", "prores_ks",
        "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le",
        output_path,
    ])
    print(f"  Saved: {output_path}  [ProRes 4444]")
    print("  Note: ProRes 4444 files are large. For web delivery, compress in")
    print("        Final Cut Pro or Compressor after verifying the key looks right.")


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

_force_cpu = False  # set by --no-gpu


def _has_encoder(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-encoders"], capture_output=True,
                             text=True, timeout=10).stdout
        return name in out
    except Exception:
        return False


def _ffmpeg(args: list[str], check: bool = True) -> bool:
    cmd = ["ffmpeg", "-y"] + args
    try:
        r = subprocess.run(cmd, check=check, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
        return r.returncode == 0
    except subprocess.CalledProcessError as e:
        if check:
            sys.exit(f"FFmpeg error:\n{e.stderr.decode()}")
        return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process(input_path, hex_color, base_output, tolerance, softness,
            webm_only, hevc_only, no_gpu):

    global _force_cpu
    _force_cpu = no_gpu

    if not shutil.which("ffmpeg"):
        sys.exit("Error: FFmpeg not found. Install from https://ffmpeg.org or via brew/apt.")

    gpu       = GPUInfo(force_cpu=no_gpu)
    webm_path = f"{base_output}.webm"
    mov_path  = f"{base_output}.mov"

    print(f"Input:      {input_path}")
    print(f"Background: {hex_color}  tolerance={tolerance}  softness={softness}")
    print(f"Keying GPU: {gpu.summary()}")
    if not hevc_only: print(f"WebM out:   {webm_path}")
    if not webm_only: print(f"Safari out: {mov_path}")

    with tempfile.TemporaryDirectory(prefix="chromakey_") as tmp_dir:
        fps, w, h, n = extract_frames(
            input_path, hex_color, tolerance, softness, tmp_dir, gpu)
        if n == 0:
            sys.exit("Error: No frames extracted.")

        if not hevc_only:
            encode_webm(tmp_dir, webm_path, fps, gpu)
        if not webm_only:
            encode_safari_mov(tmp_dir, mov_path, fps, gpu)

    print("\nDone!")
    if not hevc_only: print(f"  WebM   -> {webm_path}   (Chrome, Firefox, Edge)")
    if not webm_only: print(f"  Safari -> {mov_path}   (Safari macOS/iOS)")
    print()
    print("HTML snippet:")
    print("  <video autoplay muted playsinline loop>")
    if not webm_only:
        print(f'    <source src="{os.path.basename(mov_path)}" type="video/mp4; codecs=hvc1,ap4h">')
    if not hevc_only:
        print(f'    <source src="{os.path.basename(webm_path)}" type="video/webm">')
    print("  </video>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input",  help="Input MP4 file")
    p.add_argument("color",  help="Background hex color, e.g. '#00FF00'")
    p.add_argument("--output",    default=None,
                   help="Output base name (default: <input>_keyed). Appends .webm / .mov")
    p.add_argument("--tolerance", type=int, default=30,
                   help="Hue tolerance 0-90 (default: 30)")
    p.add_argument("--softness",  type=int, default=0,
                   help="Edge blur radius (default: 0). Try 2-5 for soft edges.")
    p.add_argument("--webm-only", action="store_true", help="Skip Safari MOV output")
    p.add_argument("--hevc-only", action="store_true", help="Skip WebM output")
    p.add_argument("--no-gpu",    action="store_true", help="Disable GPU acceleration")
    args = p.parse_args()

    if args.webm_only and args.hevc_only:
        sys.exit("Error: --webm-only and --hevc-only are mutually exclusive.")
    try:
        hex_to_bgr(args.color)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    base = args.output or args.input.rsplit(".", 1)[0] + "_keyed"
    process(args.input, args.color, base, args.tolerance, args.softness,
            args.webm_only, args.hevc_only, args.no_gpu)


if __name__ == "__main__":
    main()