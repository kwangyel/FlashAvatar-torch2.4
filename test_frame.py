import os
import sys
import random
import argparse

import cv2
import numpy as np
import torch

from scene import GaussianModel, Scene_frames
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_training_checkpoint(path, map_location=None):
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render FlashAvatar from .frame files only")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory containing MICA .frame checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained FlashAvatar checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path. Defaults to <frames_dir>/render.avi.",
    )
    parser.add_argument("--image_res", type=int, default=512, help="Output video resolution.")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS.")
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for FlashAvatar rendering. "
            "Select a GPU instance and confirm `torch.cuda.is_available()` is True."
        )

    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)
    set_random_seed(args.seed)

    frames_dir = os.path.abspath(args.frames_dir)
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(frames_dir, "render.avi")
    else:
        output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    scene = Scene_frames(frames_dir, device=args.device)

    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    model_params, gauss_params, _first_iter = _load_training_checkpoint(args.checkpoint)
    DeformModel.restore(model_params)
    gaussians.restore(gauss_params, opt, device=args.device)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        args.fps,
        (args.image_res, args.image_res),
        True,
    )

    viewpoint = scene.getCameras().copy()
    codedict = {"shape": scene.shape_param.to(args.device)}
    DeformModel.example_init(codedict)

    for viewpoint_cam in viewpoint:
        codedict["expr"] = viewpoint_cam.exp_param
        codedict["eyes_pose"] = viewpoint_cam.eyes_pose
        codedict["eyelids"] = viewpoint_cam.eyelids
        codedict["jaw_pose"] = viewpoint_cam.jaw_pose
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image = render_pkg["render"].clamp(0, 1)

        image_np = (image * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        if image_np.shape[0] != args.image_res or image_np.shape[1] != args.image_res:
            image_np = cv2.resize(image_np, (args.image_res, args.image_res), interpolation=cv2.INTER_AREA)
        image_np = image_np[:, :, [2, 1, 0]]
        out.write(image_np)

    out.release()
    print(f"Saved render video to {output_path}")
