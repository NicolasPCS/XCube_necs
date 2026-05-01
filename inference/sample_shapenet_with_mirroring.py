import os

import pytorch_lightning as pl
import argparse
import importlib
import torch
import numpy as np
import pickle
from loguru import logger 
import random
from fvdb.nn import VDBTensor
from pathlib import Path
from datetime import datetime
import trimesh

from xcube.utils.vis_util import random_seed
from xcube.utils import exp


def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser

def create_model_from_args(config_path, ckpt_path, strict=True):
    print(f"\n[DEBUG] Cargando Config: {os.path.abspath(config_path)}")
    print(f"[DEBUG] Cargando Checkpoint: {os.path.abspath(ckpt_path)}")
    model_yaml_path = Path(config_path)
    model_args = exp.parse_config_yaml(model_yaml_path)
    net_module = importlib.import_module("xcube.models." + model_args.model).Model
    args_ckpt = Path(ckpt_path)
    assert args_ckpt.exists(), "Selected checkpoint does not exist!"
    net_model = net_module.load_from_checkpoint(args_ckpt, hparams=model_args, strict=strict)
    return net_model.eval()

def get_parser():
    parser = exp.ArgumentParserX(base_config_path='../structure-ldm/configs/default/param.yaml', parents=[get_default_parser()])
    parser.add_argument('--category', type=str, required=True, help='ShapeNet category.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--total_len', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('--batch_len', type=int, default=4, help='Number of samples to generate in each batch.')
    parser.add_argument('--ema', action='store_true', help='Whether to turn on ema option.')
    parser.add_argument('--use_ddim', action='store_true', help='Whether to use ddim during sampling.')
    parser.add_argument('--ddim_step', type=int, default=100, help='Number of steps to increase ddim.')
    parser.add_argument('--use_dpm', action='store_true', help='Whether to use dpm solver during sampling.')
    parser.add_argument('--use_karras', action='store_true', help='Whether to use karras std during sampling.')
    parser.add_argument('--solver_order', type=int, default=3, help='Order of DPM solver.')
    parser.add_argument('--extract_mesh', action='store_true', help='Whether to extract mesh.')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size used to extract mesh.')
    parser.add_argument('--original', action='store_true')
    return parser

known_args = get_parser().parse_known_args()[0]
random_seed(known_args.seed)

print(known_args)

# setup model config and path
if known_args.category == "chair":
    config_coarse = "configs/shapenet/chair/train_diffusion_16x16x16_dense.yaml"
    config_fine = "configs/shapenet/chair/train_diffusion_128x128x128_sparse.yaml"
    config_nksr = "configs/shapenet/chair/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/chair/nksr_refine/last.ckpt"

    if known_args.original:
        config_coarse = "checkpoints/configs/shapenet/chair/train_diffusion_16x16x16_dense.yaml"
        config_fine = "checkpoints/configs/shapenet/chair/train_diffusion_128x128x128_sparse.yaml"
        ckpt_coarse = "checkpoints/chair/coarse_diffusion/last.ckpt"
        ckpt_fine = "checkpoints/chair/fine_diffusion/last.ckpt"
    else:
        ckpt_coarse = "/home/ncaytuir/data-local/wandb/xcube-shapenet_from_patagon/83f13urg/checkpoints/last.ckpt"
        ckpt_fine = "/home/ncaytuir/data-local/wandb/xcube-shapenet_from_patagon/qxzae5cr/checkpoints/last.ckpt"

elif known_args.category == "car":
    config_coarse = "configs/shapenet/car/train_diffusion_16x16x16_dense.yaml"
    config_fine = "configs/shapenet/car/train_diffusion_128x128x128_sparse.yaml"
    config_nksr = "configs/shapenet/car/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/car/nksr_refine/last.ckpt"

    if known_args.original:
        config_coarse = "checkpoints/configs/shapenet/car/train_diffusion_16x16x16_dense.yaml"
        config_fine = "checkpoints/configs/shapenet/car/train_diffusion_128x128x128_sparse.yaml"
        ckpt_coarse = "checkpoints/car/coarse_diffusion/last.ckpt"
        ckpt_fine = "checkpoints/car/fine_diffusion/last.ckpt"
    else:
        ckpt_coarse = "/home/ncaytuir/data-local/wandb/xcube-shapenet_from_patagon/665wxnqj/checkpoints/last.ckpt"
        ckpt_fine = "/home/ncaytuir/data-local/wandb/xcube-shapenet_from_patagon/573u3020/checkpoints/last.ckpt"

elif known_args.category == "plane":
    config_coarse = "configs/shapenet/plane/train_diffusion_16x16x16_dense.yaml"
    config_fine = "configs/shapenet/plane/train_diffusion_128x128x128_sparse.yaml"
    config_nksr = "configs/shapenet/plane/train_nksr_refine.yaml"
    ckpt_nksr = "checkpoints/plane/nksr_refine/last.ckpt"

    if known_args.original:
        config_coarse = "checkpoints/configs/shapenet/plane/train_diffusion_16x16x16_dense.yaml"
        config_fine = "checkpoints/configs/shapenet/plane/train_diffusion_128x128x128_sparse.yaml"
        ckpt_coarse = "checkpoints/plane/coarse_diffusion/last.ckpt"
        ckpt_fine = "checkpoints/plane/fine_diffusion/last.ckpt"
    else:
        ckpt_coarse = "/home/ncaytuir/data-local/wandb/xcube-shapenet_from_patagon/2wz5l363/checkpoints/last.ckpt"
        ckpt_fine = "/home/ncaytuir/data-local/wandb/xcube-shapenet_from_patagon/b2iqdmvn/checkpoints/last.ckpt"

else:
    raise ValueError("Unknown category: %s" % known_args.category)

print("\n[CHECKPOINTS]")
print("  category:", known_args.category)
print("  original:", known_args.original)
print("  config_coarse:", config_coarse)
print("  ckpt_coarse:", ckpt_coarse)
print("  config_fine:", config_fine)
print("  ckpt_fine:", ckpt_fine)

net_model = create_model_from_args(config_coarse, ckpt_coarse).cuda()
net_model_c = create_model_from_args(config_fine, ckpt_fine).cuda()

# setup nksr
if known_args.extract_mesh:
    #import nksr
    #reconstructor = create_model_from_args(config_nksr, ckpt_nksr, strict=False).cuda()
    pass

# begin sample pcs for evaluation
logger.info(f"Sampling from XCube on {known_args.category} ...")

mode_name = "original_ckpt" if known_args.original else "mirrored_ckpt"
save_folder = f"./results/{known_args.category}_{mode_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

logger.info(f"Saving results to {save_folder}")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if known_args.extract_mesh:
    mesh_folder = os.path.join(save_folder, "mesh")
    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)

coarse_original_folder = os.path.join(save_folder, "coarse_original")
coarse_mirrored_folder = os.path.join(save_folder, "coarse_mirrored")
fine_original_folder = os.path.join(save_folder, "fine_original")
fine_mirrored_folder = os.path.join(save_folder, "fine_mirrored")

os.makedirs(coarse_original_folder, exist_ok=True)
os.makedirs(coarse_mirrored_folder, exist_ok=True)
os.makedirs(fine_original_folder, exist_ok=True)
os.makedirs(fine_mirrored_folder, exist_ok=True)

def grid_to_xyz(grid_batch, batch_idx):
    grid_b = grid_batch[batch_idx]
    return grid_b.grid_to_world(grid_b.ijk.float()).jdata.cpu().numpy()

def debug_res_grid_pair(name, res, output_x):
    print(f"\n[DEBUG PAIR] {name}")
    print("  output_x.grid.total_voxels:", output_x.grid.total_voxels)

    if hasattr(res, "normal_features") and len(res.normal_features) > 0:
        normal_vdb = res.normal_features[-1]
        print("  normal.grid.total_voxels:", normal_vdb.grid.total_voxels)
        print("  normal.feature.shape:", normal_vdb.feature.jdata.shape)

    if hasattr(res, "semantic_features") and len(res.semantic_features) > 0:
        semantic_vdb = res.semantic_features[-1]
        print("  semantic.grid.total_voxels:", semantic_vdb.grid.total_voxels)
        print("  semantic.feature.shape:", semantic_vdb.feature.jdata.shape)

with torch.no_grad():
    num_sample = 0

    while num_sample < known_args.total_len:
        logger.info("Sampling %d / %d" % (num_sample, known_args.total_len))

        # =========================
        # 1. COARSE: original + mirrored
        # =========================
        res_coarse, output_x_coarse, res_coarse_mirrored, output_x_coarse_mirrored = \
            net_model.evaluation_api_with_mirroring(
                batch_size=known_args.batch_len,
                use_ddim=known_args.use_ddim,
                ddim_step=known_args.ddim_step,
                use_dpm=known_args.use_dpm,
                solver_order=known_args.solver_order,
                use_karras=known_args.use_karras,
                use_ema=known_args.ema
            )
        debug_res_grid_pair("COARSE ORIGINAL", res_coarse, output_x_coarse)
        debug_res_grid_pair("COARSE MIRRORED", res_coarse_mirrored, output_x_coarse_mirrored)
        # =========================
        # 2. FINE ORIGINAL
        # =========================
        res, output_x = net_model_c.evaluation_api(
            grids=output_x_coarse.grid,
            use_ddim=known_args.use_ddim,
            ddim_step=known_args.ddim_step,
            use_dpm=known_args.use_dpm,
            solver_order=known_args.solver_order,
            use_karras=known_args.use_karras,
            use_ema=known_args.ema,
            res_coarse=res_coarse,
        )

        # =========================
        # 3. FINE MIRRORED
        # =========================
        res_mirrored, output_x_mirrored = net_model_c.evaluation_api(
            grids=output_x_coarse_mirrored.grid,
            use_ddim=known_args.use_ddim,
            ddim_step=known_args.ddim_step,
            use_dpm=known_args.use_dpm,
            solver_order=known_args.solver_order,
            use_karras=known_args.use_karras,
            use_ema=known_args.ema,
            res_coarse=res_coarse_mirrored,
        )

        # =========================
        # 4. SAVE
        # =========================
        batch_count = output_x.grid.grid_count

        for batch_idx in range(batch_count):
            if num_sample >= known_args.total_len:
                break

            coarse_original_xyz = grid_to_xyz(output_x_coarse.grid, batch_idx)
            coarse_mirrored_xyz = grid_to_xyz(output_x_coarse_mirrored.grid, batch_idx)
            fine_original_xyz = grid_to_xyz(output_x.grid, batch_idx)
            fine_mirrored_xyz = grid_to_xyz(output_x_mirrored.grid, batch_idx)

            coarse_original_path = os.path.join(
                coarse_original_folder,
                "pc%d.npy" % num_sample
            )
            coarse_mirrored_path = os.path.join(
                coarse_mirrored_folder,
                "pc%d.npy" % num_sample
            )
            fine_original_path = os.path.join(
                fine_original_folder,
                "pc%d.npy" % num_sample
            )
            fine_mirrored_path = os.path.join(
                fine_mirrored_folder,
                "pc%d.npy" % num_sample
            )

            np.save(coarse_original_path, coarse_original_xyz)
            np.save(coarse_mirrored_path, coarse_mirrored_xyz)
            np.save(fine_original_path, fine_original_xyz)
            np.save(fine_mirrored_path, fine_mirrored_xyz)

            num_sample += 1