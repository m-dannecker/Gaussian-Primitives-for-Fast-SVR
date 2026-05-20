"""
Training entry point for Gaussian Slice-to-Volume Reconstruction (GSVR).

Loads input MRI stacks, initialises the GaussianSVR model, and optimises it
end-to-end via gradient descent.  A dense 3D reconstruction is saved to disk
at the end of training.
"""

import argparse
import torch
import numpy as np
import random
import os
import time
import yaml
import faiss
import faiss.contrib.torch_utils
from utils import *
from gsvr import GaussianSVR
import nibabel as nib


def train(cfg_data, cfg_gsvr, output_root, cfg_vis=None):
    """
    Main training loop.

    Args:
        cfg_data:     Data config dict (preprocessing, subject paths, etc.).
        cfg_gsvr:     GSVR model config dict (architecture, learning rates, etc.).
        output_root:  Directory where reconstruction outputs are saved.
        cfg_vis:      Optional visualization config dict; when its 'snapshots.enabled'
                      is true, intermediate reconstructions are saved every
                      'interval_seconds' wall-clock seconds during training.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(output_root), exist_ok=True)

    # --- 1. Data Setup ---
    [stack_imgs_nrmd, affines, slice_idcs, slice_centers_nrmd,
    coords_nrmd, values_nrmd, cov_psf, bbox_world, bbox_world_size,
    shape_reconstruction, n_slices_global, min_value_stack, max_value_stack
    ] = load_data(cfg_data)

    bbox_world = torch.from_numpy(np.array(bbox_world)).to(dtype=torch.float32, device=device)
    slice_idcs = torch.from_numpy(slice_idcs).long().to(dtype=torch.long, device=device)
    slice_centers_nrmd = torch.from_numpy(slice_centers_nrmd).to(dtype=torch.float32, device=device)
    coords_nrmd = torch.from_numpy(coords_nrmd).to(dtype=torch.float32, device=device).contiguous()
    values_nrmd = torch.from_numpy(values_nrmd).to(dtype=torch.float32, device=device)
    cov_psf = torch.from_numpy(cov_psf).to(dtype=torch.float32, device=device)
    
    # --- Pre-compute Visualization Grid ---
    vis_grid_flat_3D = create_vis_grid(shape_reconstruction, bbox_world, device)
    print(f"Number of non-zero voxels in the raw data: {len(values_nrmd)}")

    # --- 2. Gaussian Sice-to-Volume Reconstruction Model Setup ---
    gsvr = GaussianSVR(D=3, stack_imgs_nrmd=stack_imgs_nrmd, affines=affines, 
                       bbox_world=bbox_world, num_slices=n_slices_global, 
                       cfg_gsvr=cfg_gsvr, device=device)
    gsvr.to(device)
    optimizer, scheduler, grad_scaler = init_optim(gsvr, cfg_gsvr, cfg_gsvr['max_epochs'])
    gpu_topk = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, faiss.GpuIndexFlatConfig())
    topK_every = cfg_gsvr['g_primitives']['topK_every']
    K = cfg_gsvr['g_primitives']['K_neighbors']
    print(f"Training on device: {device}")

    # --- 6. Training Loop ---
    psf = cfg_gsvr['psf']
    max_epochs = cfg_gsvr['max_epochs']
    slice_scaling = cfg_gsvr['slice_scaling']
    slice_weighting = cfg_gsvr['slice_weighting']
    lambda_reg = cfg_gsvr['g_primitives']['lambda_reg']
    lambda_color_tv = cfg_gsvr['g_primitives'].get('lambda_color_tv', 0.0)
    K_color = cfg_gsvr['g_primitives'].get('K_color', 8)
    lambda_mc_rot = cfg_gsvr.get('lambda_mc_rot', 0.0)
    lambda_mc_trans = cfg_gsvr.get('lambda_mc_trans', 0.0)
    eta_min_factor = cfg_gsvr.get('lr_eta_min_factor', 0.01)
    scale_target = cfg_gsvr['g_primitives']['scale_target']
    warmup_epochs = max_epochs // 4
    total_ttime = 0
    topK_idcs = None
    g2g_knn_idcs = None    # (num_active, K_color) — rebuilt with FAISS at topK_every cadence
    num_coords = coords_nrmd.shape[0]

    # --- Coarse-to-fine setup ---
    c2f_cfg = cfg_gsvr.get('coarse_to_fine', {})
    c2f_enabled = c2f_cfg.get('enabled', False)
    if c2f_enabled:
        num_gaussians_max = cfg_gsvr['g_primitives']['num_gaussians']
        c2f_alpha = c2f_cfg.get('scale_target_alpha', 0.3)
        # Build epoch → num_active lookup; schedule entries are [epoch, num_active]
        growth_schedule = {int(ep): int(n) for ep, n in c2f_cfg['growth_schedule']}
        # Per-Gaussian error buffer for splitting decisions (updated at topK_every epochs)
        c2f_error_per_gaussian = torch.zeros(num_gaussians_max, device=device)
        c2f_error_counts = torch.zeros(num_gaussians_max, device=device)
        print(f"Coarse-to-fine enabled: starting with "
              f"{gsvr.gaussian_primitives.num_active.item()} Gaussians, "
              f"max {num_gaussians_max}")

    # --- Mini-batch setup ---
    mini_batch_size = cfg_gsvr.get('mini_batch_size', None)
    if mini_batch_size is not None:
        B = int(mini_batch_size)
        num_chunks = num_coords // B
        assert num_chunks >= 1, f"mini_batch_size {B} must be <= num_coords {num_coords}"
    else:
        B = num_coords
        num_chunks = 1

    # torch.compile mode selection:
    # - "reduce-overhead" uses CUDA graphs for maximum speed but requires a
    #   single forward/backward per step (static buffer reuse).
    # - "default" uses kernel fusion without CUDA graphs — compatible with
    #   gradient accumulation (multiple forward passes before optimizer.step).
    if num_chunks > 1:
        gsvr = torch.compile(gsvr, mode="default")
    else:
        gsvr = torch.compile(gsvr, mode="reduce-overhead")

    # --- Snapshot visualization setup (opt-in) ---
    vis_enabled = bool(cfg_vis and cfg_vis.get('snapshots', {}).get('enabled', False))
    if vis_enabled:
        snap_cfg = cfg_vis['snapshots']
        snapshot_dir = os.path.join(output_root, snap_cfg.get('subdir', 'snapshots'))
        os.makedirs(snapshot_dir, exist_ok=True)
        vis_interval = float(snap_cfg.get('interval_seconds', 10))


    t0 = time.time()
    for i in range(max_epochs):
        if vis_enabled and (i < max_epochs - 1):
            if i == 0:
                vis_wall_t0 = time.time()
                vis_last_t = vis_wall_t0
                vis_total_render_time = 0.0  # cumulative time spent in snapshot renders, excluded from training time
            now = time.time()
            if now - vis_last_t >= vis_interval or (i == 0):
                visualize_gaussians(
                    gsvr, gpu_topk, K, vis_grid_flat_3D, shape_reconstruction,
                    cfg_data, snapshot_dir, bbox_world,
                    min_value_stack, max_value_stack,
                    coords_nrmd, slice_centers_nrmd, slice_idcs,
                    ttime=now - vis_wall_t0 - vis_total_render_time, device=device,
                )
                # Schedule next snapshot from post-render time, not pre-render,
                # otherwise a slow render (>= interval_seconds) re-triggers every epoch.
                vis_last_t = time.time()
                # Exclude snapshot rendering from training-time accounting:
                # advance both t0 (per-50-epoch print delta) and the wall-clock
                # offset used for the next snapshot's ttime label.
                render_dt = vis_last_t - now
                t0 += render_dt
                vis_total_render_time += render_dt

        # --- Epoch start: build permuted index ---
        # Full-batch: sequential order (no shuffling, identical to original behaviour).
        # Mini-batch: random permutation so each chunk has a representative spatial/slice mix.
        if mini_batch_size is not None:
            perm = torch.randperm(num_coords, device=device)[:num_chunks * B]
        else:
            perm = torch.arange(num_coords, device=device)

        # --- Coarse-to-fine: growth step (split & activate new Gaussians) ---
        if c2f_enabled and i in growth_schedule:
            new_active = growth_schedule[i]
            cur_active = gsvr.gaussian_primitives.num_active.item()
            if new_active > cur_active:
                # Compute per-Gaussian mean error for splitting decisions
                if c2f_error_counts[:cur_active].sum() > 0:
                    safe_counts = c2f_error_counts[:cur_active].clamp(min=1)
                    error_signal = c2f_error_per_gaussian[:cur_active] / safe_counts
                else:
                    error_signal = None
                gsvr.gaussian_primitives.split_and_activate(new_active, error_per_gaussian=error_signal)
                # Reset error buffers for the next phase
                c2f_error_per_gaussian.zero_()
                c2f_error_counts.zero_()
                # Warm restart: reset LRs and create fresh cosine schedule for next phase
                next_growth_epochs = [ep for ep in sorted(growth_schedule) if ep > i]
                T_max_next = (next_growth_epochs[0] - i) if next_growth_epochs else (max_epochs - i)
                scheduler = restart_scheduler(optimizer, T_max_next, eta_min_factor)

        # --- TopK refresh (every topK_every epochs) ---
        # Compute for ALL coordinates (not permuted) so topK_idcs[i] always
        # corresponds to coords_nrmd[i], regardless of the epoch's permutation.
        if i % topK_every == 0:
            num_active = gsvr.gaussian_primitives.num_active.item()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type):
                    coords_tf_topk, _ = gsvr.motion_correction(
                        coords_nrmd, slice_centers_nrmd, slice_idcs, None)
            topK_idcs = topK_neighbors(gpu_topk, gsvr.gaussian_primitives.mu.data[:num_active], K, coords_tf_topk)
            # Graph-TV on colour: K-NN among active Gaussian centres. Same FAISS
            # index, stale-by-design between rebuilds (matches topK_idcs cadence).
            # K_color+1 then drop column 0 to exclude self (distance 0 to itself).
            if lambda_color_tv > 0.0:
                mu_active = gsvr.gaussian_primitives.mu.data[:num_active]
                g2g_knn_full = topK_neighbors(gpu_topk, mu_active, K_color + 1, mu_active)
                g2g_knn_idcs = g2g_knn_full[:, 1:]

        # --- Gradient accumulation loop ---
        loss_history = {'sr': [], 'sr_reg': [], 'mc_rot': [], 'mc_trans': [], 'color_tv': []}
        optimizer.zero_grad()
        run_outlier_update = slice_weighting and i >= warmup_epochs and i % topK_every == 0
        outlier_res_parts  = [] if run_outlier_update else None  # brain residuals per chunk
        outlier_ids_parts  = [] if run_outlier_update else None  # slice IDs per chunk

        for c in range(num_chunks):
            cidx = slice(c * B, (c + 1) * B)
            perm_c        = perm[cidx]
            b_values_c    = values_nrmd[perm_c]
            b_slice_ids_c = slice_idcs[perm_c]

            with torch.amp.autocast(device_type=device.type):
                coords_tf_c, cov_psf_tf_c = gsvr.motion_correction(
                    coords_nrmd[perm_c], slice_centers_nrmd[perm_c],
                    slice_idcs[perm_c], cov_psf if psf else None)
                values_pred_c = gsvr(coords_tf_c, topK_idcs[perm_c], cov_psf_tf_c)

                # Per-slice intensity scaling (learnable): corrects physical
                # inter-slice intensity variation (B1 field, excitation profile)
                if slice_scaling:
                    ss = torch.nn.functional.softplus(gsvr.slice_scales)
                    # Clamp denominator: if many slices' softplus values collapse
                    # toward zero (e.g. after a pathological outlier update),
                    # ss.mean() can shrink to ~0 and produce inf/NaN scales.
                    ss = ss / ss.mean().clamp(min=1e-3)
                    values_pred_c = values_pred_c * ss[b_slice_ids_c]

                # Collect residuals for outlier detection (detached, no grad impact).
                # All supervisory voxels are brain by construction (no halo path).
                if run_outlier_update:
                    with torch.no_grad():
                        res_c = (values_pred_c.detach() - b_values_c).abs()
                        outlier_res_parts.append(res_c)
                        outlier_ids_parts.append(b_slice_ids_c)

                # Coarse-to-fine: accumulate per-Gaussian error for split decisions.
                # Scatter each voxel's residual to its nearest (first) Gaussian.
                # Runs at topK_every epochs (same cadence as FAISS rebuild).
                if c2f_enabled and i % topK_every == 0:
                    with torch.no_grad():
                        res_c2f = (values_pred_c.detach() - b_values_c).abs()
                        nearest_g = topK_idcs[perm_c, 0]  # (B,) index of closest Gaussian
                        c2f_error_per_gaussian.scatter_add_(0, nearest_g, res_c2f)
                        c2f_error_counts.scatter_add_(0, nearest_g, torch.ones_like(res_c2f))

                # Per-voxel loss weights: per-slice Welsch outlier weights only
                # (halo path removed — output masking is done by build_recon_mask
                # at render time).
                if slice_weighting:
                    voxel_weights_c = gsvr.slice_outlier_weights[b_slice_ids_c]
                else:
                    voxel_weights_c = None

                # Coarse-to-fine: inflate scale target and restrict regularisation
                # to active Gaussians only
                if c2f_enabled:
                    cur_active = gsvr.gaussian_primitives.num_active.item()
                    st_eff = compute_adaptive_scale_target(
                        scale_target, cur_active, num_gaussians_max, c2f_alpha)
                else:
                    st_eff = scale_target
                    cur_active = None

                loss_c, loss_dict_c = loss_composition(
                    values_pred_c, b_values_c, gsvr.rotation_mc, gsvr.translation_mc,
                    lambda_reg, st_eff, gsvr.gaussian_primitives.scaling,
                    voxel_weights=voxel_weights_c, num_active=cur_active,
                    lambda_mc_rot=lambda_mc_rot, lambda_mc_trans=lambda_mc_trans,
                    color=gsvr.gaussian_primitives.color,
                    g2g_knn_idcs=g2g_knn_idcs,
                    lambda_color_tv=lambda_color_tv)
                loss_c = loss_c / num_chunks   # normalise so accumulated grad == full-batch grad

            loss_history['sr'].append(loss_dict_c['sr'])
            loss_history['sr_reg'].append(loss_dict_c['sr_reg'])
            loss_history['mc_rot'].append(loss_dict_c['mc_rot'])
            loss_history['mc_trans'].append(loss_dict_c['mc_trans'])
            loss_history['color_tv'].append(loss_dict_c['color_tv'])
            grad_scaler.scale(loss_c).backward()

        # grad_scaler.step internally unscales accumulated gradients once, then updates
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # --- Update outlier weights from residuals collected above ---
        if run_outlier_update:
            with torch.no_grad():
                all_res = torch.cat(outlier_res_parts)
                all_ids = torch.cat(outlier_ids_parts)
                weights, active_slices, oh_info = compute_slice_outlier_weights(
                    all_res, all_ids, n_slices_global)
                n_active = int(active_slices.sum().item())
                n_low = oh_info['n_below_0.1']
                # Catastrophic-cascade guard: if a single update would push the
                # majority of active slices below 0.1 weight, the robust scale
                # has collapsed (residuals tight after convergence, MAD → floor,
                # natural anatomy variance dominates). Skip the update; previous
                # weights remain in effect.
                if n_active > 0 and n_low / n_active > 0.5:
                    print(f"  Outlier update skipped: would suppress "
                          f"{n_low}/{n_active} slices below 0.1 weight "
                          f"(med={oh_info['med']:.4f}, "
                          f"mad_eff={oh_info['mad_eff']:.4f}, "
                          f"z_max={oh_info['z_max']:.2f})")
                else:
                    gsvr.slice_outlier_weights.copy_(weights)
                    n_downweighted = int(((weights < 0.5) & active_slices).sum().item())
                    print(f"  Outlier detection ({n_active} active slices): "
                          f"{n_downweighted} down-weighted, {n_low} below 0.1 "
                          f"(med={oh_info['med']:.4f}, "
                          f"mad_eff={oh_info['mad_eff']:.4f}, "
                          f"z_max={oh_info['z_max']:.2f})")

        loss_mean = {'sr': np.array(loss_history['sr']).mean(),
                     'sr_reg': np.array(loss_history['sr_reg']).mean(),
                     'mc_rot': np.array(loss_history['mc_rot']).mean(),
                     'mc_trans': np.array(loss_history['mc_trans']).mean(),
                     'color_tv': np.array(loss_history['color_tv']).mean()}

        if scheduler is not None: scheduler.step() # Step scheduler once per epoch

        if i % 50 == 0 or (i == max_epochs-1):
            t1 = time.time()
            t_delta = t1-t0
            total_ttime += t_delta
            c2f_info = (f', active_gaussians: {gsvr.gaussian_primitives.num_active.item()}'
                        if c2f_enabled else '')
            if slice_scaling:
                with torch.no_grad():
                    ss_dbg = torch.nn.functional.softplus(gsvr.slice_scales)
                    ss_dbg = ss_dbg / ss_dbg.mean().clamp(min=1e-3)
                ss_info = (f', slice_scale min: {ss_dbg.min().item():.3f}, '
                           f'p5: {ss_dbg.quantile(0.05).item():.3f}, '
                           f'max: {ss_dbg.max().item():.3f}, '
                           f'n<0.2: {int((ss_dbg < 0.2).sum().item())}')
            else:
                ss_info = ''
            ctv_info = (f', loss color_tv: {loss_mean["color_tv"]:.6f}'
                        if lambda_color_tv > 0.0 else '')
            print(f'Epoch {i} loss sr: {loss_mean["sr"]:.6f} ',
                  f'loss sr_reg: {loss_mean["sr_reg"]:.6f}, '
                  f'loss mc_rot: {loss_mean["mc_rot"]:.6f}, '
                  f'loss mc_trans: {loss_mean["mc_trans"]:.6f}'
                  f'{ctv_info}, '
                  f'time per epoch: {t_delta/50:.2f}s, '
                  f'total training time: {total_ttime:.2f}s{c2f_info}{ss_info}')
            t0 = time.time()

        if (i==max_epochs-1): # or (i % 200 == 0) and (i > 0):
            print(f"Visualizing epoch {i}...")
            visualize_gaussians(gsvr, gpu_topk, K, vis_grid_flat_3D, shape_reconstruction, cfg_data, output_root, bbox_world,
                                min_value_stack, max_value_stack,
                                coords_nrmd, slice_centers_nrmd, slice_idcs,
                                ttime=total_ttime, device=device)
            t0 = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./GS_SVR/configs/config_subjects_real.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    output_root = os.path.join(config['experiment']['output_root'], config['experiment']['name'], config['data']['subject']['name'])
    os.makedirs(output_root, exist_ok=True)
    train(cfg_data=config['data'], cfg_gsvr=config['gsvr'], output_root=output_root,
          cfg_vis=config.get('visualization'))
    print("Training completed successfully.")