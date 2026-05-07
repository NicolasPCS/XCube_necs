import os

# Debe ir antes de importar torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import argparse
import importlib
from pathlib import Path
from datetime import datetime

import pytorch_lightning as pl
import torch
import numpy as np
from loguru import logger

from xcube.utils.vis_util import random_seed
from xcube.utils import exp


# ============================================================
# PATCH EMA
# ============================================================

def patch_litema_copy_to():
    """
    Corrige el bug:
        assert not key in self.m_name2s_name

    Para usar EMA debe ser:
        assert key in self.m_name2s_name

    Esto evita modificar manualmente:
        xcube/modules/diffusionmodules/ema.py
    """
    try:
        import xcube.modules.diffusionmodules.ema as ema_mod

        def copy_to_fixed(self, model):
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    assert key in self.m_name2s_name, f"EMA key not found: {key}"
                    s_name = self.m_name2s_name[key]
                    assert s_name in shadow_params, f"EMA shadow buffer not found: {s_name}"
                    m_param[key].data.copy_(shadow_params[s_name].data)

        ema_mod.LitEma.copy_to = copy_to_fixed
        logger.info("[PATCH] LitEma.copy_to patched successfully.")

    except Exception as e:
        logger.warning(f"[PATCH] Could not patch LitEma.copy_to: {e}")


patch_litema_copy_to()


# ============================================================
# CUDA UTILS
# ============================================================

def cuda_cleanup(tag=""):
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if tag:
        logger.info(f"[CUDA CLEANUP] {tag}")


def cuda_mem(tag=""):
    if not torch.cuda.is_available():
        return

    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    logger.info(
        f"[CUDA MEM] {tag} | "
        f"free={free / 1024**3:.2f}GB | "
        f"total={total / 1024**3:.2f}GB | "
        f"allocated={allocated / 1024**3:.2f}GB | "
        f"reserved={reserved / 1024**3:.2f}GB"
    )


def is_cuda_oom(error):
    msg = str(error).lower()
    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "cuda out of memory" in msg
        or "out of memory" in msg
    )


def move_model(model, device, name):
    logger.info(f"[MODEL] Moving {name} to {device}")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    cuda_cleanup(f"after moving {name} to {device}")
    return model


# ============================================================
# PARSER
# ============================================================

def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser


def get_parser():
    parser = exp.ArgumentParserX(
        base_config_path="../structure-ldm/configs/default/param.yaml",
        parents=[get_default_parser()]
    )

    parser.add_argument("--category", type=str, required=True, choices=["chair", "car", "plane"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_len", type=int, default=1000)
    parser.add_argument("--batch_len", type=int, default=1)

    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--ddim_step", type=int, default=100)
    parser.add_argument("--use_dpm", action="store_true")
    parser.add_argument("--use_karras", action="store_true")
    parser.add_argument("--solver_order", type=int, default=3)

    parser.add_argument("--extract_mesh", action="store_true")
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--original", action="store_true")

    parser.add_argument("--max_oom_retries", type=int, default=3)

    return parser


# ============================================================
# MODEL LOADING
# ============================================================

def create_model_from_args(config_path, ckpt_path, strict=True):
    print(f"\n[DEBUG] Cargando Config: {os.path.abspath(config_path)}")
    print(f"[DEBUG] Cargando Checkpoint: {os.path.abspath(ckpt_path)}")

    model_yaml_path = Path(config_path)
    args_ckpt = Path(ckpt_path)

    assert model_yaml_path.exists(), f"Config does not exist: {model_yaml_path}"
    assert args_ckpt.exists(), f"Selected checkpoint does not exist: {args_ckpt}"

    model_args = exp.parse_config_yaml(model_yaml_path)
    net_module = importlib.import_module("xcube.models." + model_args.model).Model

    # Importante: cargar en CPU para no llenar GPU al inicio.
    net_model = net_module.load_from_checkpoint(
        args_ckpt,
        hparams=model_args,
        strict=strict,
        map_location="cpu"
    )

    net_model = net_model.cpu()
    net_model.eval()
    net_model.requires_grad_(False)

    return net_model


# ============================================================
# PATHS
# ============================================================

def get_paths(category, original):
    if category == "chair":
        config_coarse = "configs/shapenet/chair/train_diffusion_16x16x16_dense.yaml"
        config_fine = "configs/shapenet/chair/train_diffusion_128x128x128_sparse.yaml"
        config_nksr = "configs/shapenet/chair/train_nksr_refine.yaml"
        ckpt_nksr = "checkpoints/chair/nksr_refine/last.ckpt"

        if original:
            ckpt_coarse = "/home/isipiran/XCube_necs/checkpoints/chair(original)/coarse_diffusion/last.ckpt"
            ckpt_fine = "/home/isipiran/XCube_necs/checkpoints/chair(original)/fine_diffusion/last.ckpt"
        else:
            ckpt_coarse = "/home/isipiran/wandb/xcube-shapenet/83f13urg/checkpoints/epoch=000109-step=000052750.ckpt"
            ckpt_fine = "/home/isipiran/wandb/xcube-shapenet/qxzae5cr/checkpoints/epoch=000101-step=000024540.ckpt"

    elif category == "car":
        config_coarse = "configs/shapenet/car/train_diffusion_16x16x16_dense.yaml"
        config_fine = "configs/shapenet/car/train_diffusion_128x128x128_sparse.yaml"
        config_nksr = "configs/shapenet/car/train_nksr_refine.yaml"
        ckpt_nksr = "checkpoints/car/nksr_refine/last.ckpt"

        if original:
            ckpt_coarse = "/home/isipiran/XCube_necs/checkpoints/car(original)/coarse_diffusion/last.ckpt"
            ckpt_fine = "/home/isipiran/XCube_necs/checkpoints/car(original)/fine_diffusion/last.ckpt"
        else:
            ckpt_coarse = "/home/isipiran/wandb/xcube-shapenet/665wxnqj/checkpoints/epoch=000109-step=000042100.ckpt"
            ckpt_fine = "/home/isipiran/wandb/xcube-shapenet/573u3020/checkpoints/epoch=000117-step=000009060.ckpt"

    elif category == "plane":
        config_coarse = "configs/shapenet/plane/train_diffusion_16x16x16_dense.yaml"
        config_fine = "configs/shapenet/plane/train_diffusion_128x128x128_sparse.yaml"
        config_nksr = "configs/shapenet/plane/train_nksr_refine.yaml"
        ckpt_nksr = "checkpoints/plane/nksr_refine/last.ckpt"

        if original:
            ckpt_coarse = "/home/isipiran/XCube_necs/checkpoints/plane(original)/coarse_diffusion/last.ckpt"
            ckpt_fine = "/home/isipiran/XCube_necs/checkpoints/plane(original)/fine_diffusion/last.ckpt"
        else:
            ckpt_coarse = "/home/isipiran/wandb/xcube-shapenet/2wz5l363/checkpoints/epoch=000099-step=000044050.ckpt"
            ckpt_fine = "/home/isipiran/wandb/xcube-shapenet/b2iqdmvn/checkpoints/epoch=000117-step=000026100.ckpt"

    else:
        raise ValueError(f"Unknown category: {category}")

    return config_coarse, ckpt_coarse, config_fine, ckpt_fine, config_nksr, ckpt_nksr


# ============================================================
# GRID / SAVE
# ============================================================

def grid_to_xyz(grid_batch, batch_idx):
    grid_b = grid_batch[batch_idx]
    xyz = grid_b.grid_to_world(grid_b.ijk.float()).jdata
    return xyz.detach().cpu().numpy()


def debug_res_grid_pair(name, res, output_x):
    print(f"\n[DEBUG PAIR] {name}")
    print("  output_x.grid.total_voxels:", output_x.grid.total_voxels)
    print("  output_x.grid.grid_count:", output_x.grid.grid_count)

    if hasattr(res, "normal_features") and len(res.normal_features) > 0:
        normal_vdb = res.normal_features[-1]
        print("  normal.grid.total_voxels:", normal_vdb.grid.total_voxels)
        print("  normal.feature.shape:", normal_vdb.feature.jdata.shape)

    if hasattr(res, "semantic_features") and len(res.semantic_features) > 0:
        semantic_vdb = res.semantic_features[-1]
        print("  semantic.grid.total_voxels:", semantic_vdb.grid.total_voxels)
        print("  semantic.feature.shape:", semantic_vdb.feature.jdata.shape)


def save_batch(
    sample_ids,
    coarse_original_list,
    coarse_mirrored_list,
    fine_original_list,
    fine_mirrored_list,
    save_folders
):
    for i, sample_id in enumerate(sample_ids):
        np.save(
            os.path.join(save_folders["coarse_original"], f"pc{sample_id}.npy"),
            coarse_original_list[i]
        )

        np.save(
            os.path.join(save_folders["coarse_mirrored"], f"pc{sample_id}.npy"),
            coarse_mirrored_list[i]
        )

        np.save(
            os.path.join(save_folders["fine_original"], f"pc{sample_id}.npy"),
            fine_original_list[i]
        )

        np.save(
            os.path.join(save_folders["fine_mirrored"], f"pc{sample_id}.npy"),
            fine_mirrored_list[i]
        )


# ============================================================
# BATCH GENERATION
# ============================================================

def generate_batch(
    net_model,
    net_model_c,
    args,
    start_sample,
    batch_size,
    save_folders
):
    device = "cuda"
    actual_bs = min(batch_size, args.total_len - start_sample)
    sample_ids = list(range(start_sample, start_sample + actual_bs))

    coarse_original_list = []
    coarse_mirrored_list = []
    fine_original_list = []
    fine_mirrored_list = []

    res_coarse = None
    output_x_coarse = None
    res_coarse_mirrored = None
    output_x_coarse_mirrored = None
    res_fine = None
    output_x_fine = None
    res_fine_mirrored = None
    output_x_fine_mirrored = None

    try:
        # ====================================================
        # 1. COARSE MODEL EN GPU
        # ====================================================
        cuda_mem("before coarse")
        net_model = move_model(net_model, device, "coarse_model")

        with torch.inference_mode():
            res_coarse, output_x_coarse, res_coarse_mirrored, output_x_coarse_mirrored = \
                net_model.evaluation_api_with_mirroring(
                    batch_size=actual_bs,
                    use_ddim=args.use_ddim,
                    ddim_step=args.ddim_step,
                    use_dpm=args.use_dpm,
                    solver_order=args.solver_order,
                    use_karras=args.use_karras,
                    use_ema=True
                )

        debug_res_grid_pair("COARSE ORIGINAL", res_coarse, output_x_coarse)
        debug_res_grid_pair("COARSE MIRRORED", res_coarse_mirrored, output_x_coarse_mirrored)

        # Sacamos coarse model de GPU, pero dejamos vivos los resultados coarse.
        net_model = move_model(net_model, "cpu", "coarse_model")
        cuda_mem("after coarse model to CPU")

        # ====================================================
        # 2. FINE MODEL EN GPU - ORIGINAL
        # ====================================================
        net_model_c = move_model(net_model_c, device, "fine_model")

        with torch.inference_mode():
            res_fine, output_x_fine = net_model_c.evaluation_api(
                grids=output_x_coarse.grid,
                use_ddim=args.use_ddim,
                ddim_step=args.ddim_step,
                use_dpm=args.use_dpm,
                solver_order=args.solver_order,
                use_karras=args.use_karras,
                use_ema=True,
                res_coarse=res_coarse,
            )

        batch_count = min(
            len(sample_ids),
            output_x_coarse.grid.grid_count,
            output_x_coarse_mirrored.grid.grid_count,
            output_x_fine.grid.grid_count
        )

        sample_ids = sample_ids[:batch_count]

        for batch_idx in range(batch_count):
            coarse_original_list.append(grid_to_xyz(output_x_coarse.grid, batch_idx))
            coarse_mirrored_list.append(grid_to_xyz(output_x_coarse_mirrored.grid, batch_idx))
            fine_original_list.append(grid_to_xyz(output_x_fine.grid, batch_idx))

        # Liberar fine original y coarse original antes del mirrored.
        del res_fine
        del output_x_fine
        del res_coarse
        del output_x_coarse

        res_fine = None
        output_x_fine = None
        res_coarse = None
        output_x_coarse = None

        cuda_cleanup("after fine original")

        # ====================================================
        # 3. FINE MODEL EN GPU - MIRRORED
        # ====================================================
        with torch.inference_mode():
            res_fine_mirrored, output_x_fine_mirrored = net_model_c.evaluation_api(
                grids=output_x_coarse_mirrored.grid,
                use_ddim=args.use_ddim,
                ddim_step=args.ddim_step,
                use_dpm=args.use_dpm,
                solver_order=args.solver_order,
                use_karras=args.use_karras,
                use_ema=True,
                res_coarse=res_coarse_mirrored,
            )

        mirrored_count = min(
            len(sample_ids),
            output_x_fine_mirrored.grid.grid_count
        )

        sample_ids = sample_ids[:mirrored_count]
        coarse_original_list = coarse_original_list[:mirrored_count]
        coarse_mirrored_list = coarse_mirrored_list[:mirrored_count]
        fine_original_list = fine_original_list[:mirrored_count]

        for batch_idx in range(mirrored_count):
            fine_mirrored_list.append(grid_to_xyz(output_x_fine_mirrored.grid, batch_idx))

        # Liberar mirrored.
        del res_fine_mirrored
        del output_x_fine_mirrored
        del res_coarse_mirrored
        del output_x_coarse_mirrored

        res_fine_mirrored = None
        output_x_fine_mirrored = None
        res_coarse_mirrored = None
        output_x_coarse_mirrored = None

        # Sacar fine model de GPU.
        net_model_c = move_model(net_model_c, "cpu", "fine_model")

        if len(sample_ids) == 0:
            raise RuntimeError("No samples generated in this batch.")

        # ====================================================
        # 4. SAVE
        # ====================================================
        save_batch(
            sample_ids,
            coarse_original_list,
            coarse_mirrored_list,
            fine_original_list,
            fine_mirrored_list,
            save_folders
        )

        logger.info(f"[SAVE] Saved {len(sample_ids)} samples: {sample_ids[0]} to {sample_ids[-1]}")

        return len(sample_ids), net_model, net_model_c

    finally:
        try:
            net_model = net_model.to("cpu")
        except Exception:
            pass

        try:
            net_model_c = net_model_c.to("cpu")
        except Exception:
            pass

        for obj in [
            res_coarse,
            output_x_coarse,
            res_coarse_mirrored,
            output_x_coarse_mirrored,
            res_fine,
            output_x_fine,
            res_fine_mirrored,
            output_x_fine_mirrored
        ]:
            try:
                del obj
            except Exception:
                pass

        cuda_cleanup("finally generate_batch")


# ============================================================
# MAIN
# ============================================================

def main():
    args = get_parser().parse_known_args()[0]

    random_seed(args.seed)
    torch.set_grad_enabled(False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(args)

    config_coarse, ckpt_coarse, config_fine, ckpt_fine, config_nksr, ckpt_nksr = \
        get_paths(args.category, args.original)

    print("\n[CHECKPOINTS]")
    print("  category:", args.category)
    print("  original:", args.original)
    print("  config_coarse:", config_coarse)
    print("  ckpt_coarse:", ckpt_coarse)
    print("  config_fine:", config_fine)
    print("  ckpt_fine:", ckpt_fine)

    assert Path(config_coarse).exists(), f"Coarse config does not exist: {config_coarse}"
    assert Path(config_fine).exists(), f"Fine config does not exist: {config_fine}"
    assert Path(ckpt_coarse).exists(), f"Coarse checkpoint does not exist: {ckpt_coarse}"
    assert Path(ckpt_fine).exists(), f"Fine checkpoint does not exist: {ckpt_fine}"

    logger.info(f"Sampling from XCube on {args.category} with EMA...")

    mode_name = "original_ckpt" if args.original else "mirrored_ckpt"
    save_folder = f"./results/{args.category}_{mode_name}_ema_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    logger.info(f"Saving results to {save_folder}")
    os.makedirs(save_folder, exist_ok=True)

    save_folders = {
        "coarse_original": os.path.join(save_folder, "coarse_original"),
        "coarse_mirrored": os.path.join(save_folder, "coarse_mirrored"),
        "fine_original": os.path.join(save_folder, "fine_original"),
        "fine_mirrored": os.path.join(save_folder, "fine_mirrored"),
    }

    for folder in save_folders.values():
        os.makedirs(folder, exist_ok=True)

    # ========================================================
    # LOAD MODELS ONCE IN CPU
    # ========================================================
    logger.info("[LOAD] Loading coarse model in CPU")
    net_model = create_model_from_args(config_coarse, ckpt_coarse)

    logger.info("[LOAD] Loading fine model in CPU")
    net_model_c = create_model_from_args(config_fine, ckpt_fine)

    net_model = net_model.cpu()
    net_model_c = net_model_c.cpu()

    net_model.eval()
    net_model_c.eval()

    net_model.requires_grad_(False)
    net_model_c.requires_grad_(False)

    cuda_cleanup("after loading models in CPU")

    # ========================================================
    # SAMPLING LOOP
    # ========================================================
    num_sample = 0
    current_bs = max(1, args.batch_len)
    oom_retries_at_bs1 = 0

    while num_sample < args.total_len:
        current_bs = min(current_bs, args.total_len - num_sample)

        logger.info(
            f"[LOOP] Sampling {num_sample} / {args.total_len} "
            f"with batch_size={current_bs}"
        )

        try:
            saved_count, net_model, net_model_c = generate_batch(
                net_model=net_model,
                net_model_c=net_model_c,
                args=args,
                start_sample=num_sample,
                batch_size=current_bs,
                save_folders=save_folders
            )

            num_sample += saved_count
            oom_retries_at_bs1 = 0

        except Exception as e:
            if is_cuda_oom(e):
                logger.error(f"[OOM] sample={num_sample}, batch={current_bs}")
                logger.error(str(e))

                try:
                    net_model = net_model.to("cpu")
                except Exception:
                    pass

                try:
                    net_model_c = net_model_c.to("cpu")
                except Exception:
                    pass

                cuda_cleanup("after OOM")

                if current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    logger.warning(f"[OOM RECOVERY] Reducing batch_size to {current_bs}")
                    continue

                oom_retries_at_bs1 += 1

                if oom_retries_at_bs1 <= args.max_oom_retries:
                    logger.warning(
                        f"[OOM RECOVERY] batch_size=1 failed. "
                        f"Retry {oom_retries_at_bs1}/{args.max_oom_retries}"
                    )
                    continue

                report_path = os.path.join(save_folder, "OOM_REPORT.txt")
                with open(report_path, "a") as f:
                    f.write("OOM persisted with batch_size=1\n")
                    f.write(f"Stopped at sample index: {num_sample}\n")
                    f.write(f"category: {args.category}\n")
                    f.write(f"original: {args.original}\n")
                    f.write(f"error: {str(e)}\n")

                logger.error(f"[STOP] OOM persisted with batch_size=1. Report saved to {report_path}")
                break

            else:
                error_path = os.path.join(save_folder, "ERROR_REPORT.txt")
                with open(error_path, "a") as f:
                    f.write(str(e))
                    f.write("\n")

                logger.exception(f"[ERROR] Non-OOM error. Report saved to {error_path}")
                break

    try:
        net_model = net_model.to("cpu")
    except Exception:
        pass

    try:
        net_model_c = net_model_c.to("cpu")
    except Exception:
        pass

    cuda_cleanup("end")

    logger.info(f"[DONE] Generated {num_sample} samples.")
    logger.info(f"[DONE] Results saved in: {save_folder}")


if __name__ == "__main__":
    main()