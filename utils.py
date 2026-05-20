"""
Utility functions for the GSVR pipeline: data loading, PSF generation,
optimiser setup, k-NN search, loss computation, and visualisation.
"""

import numpy as np
import torch
import nibabel as nib
import os
import scipy.ndimage
import ants
import math
from torch.nn import functional as F


def generate_cov_psf(affine3x3, spacing, slice_thickness=None):
    """
    Builds the PSF covariance matrix for one MRI stack, expressed in world
    space.  The PSF is approximated as an anisotropic 3D Gaussian.

    In-plane standard deviations are derived from the voxel spacing via
    sigma = spacing * FWHM_to_sigma, where FWHM_to_sigma = 1/(2 sqrt(2 ln 2)).
    The factor of 1.2 scales the FWHM to match the scanner PSF convention
    (1.2× oversampling relative to the nominal voxel size).

    Through-plane: if slice_thickness is provided it overrides the z-spacing,
    and the factor 1.2 is removed (thickness already represents the full
    excitation profile width).

    The diagonal covariance is then rotated into world space by the slice
    orientation matrix L (unit column vectors of the affine).

    Args:
        affine3x3:       (4, 4) or (3, 3) NIfTI affine (only the 3×3 rotation/
                         scaling part is used).
        spacing:         (3,) voxel spacings in mm [dx, dy, dz].
        slice_thickness: Slice thickness in mm, or None to use spacing[2].

    Returns:
        Sigma_psf: (3, 3) world-space PSF covariance matrix.
    """
    fwhm_to_std = 1.2 / (2 * np.sqrt(2 * np.log(2)))
    sigma_psf_diag = np.array(spacing) * fwhm_to_std
    if slice_thickness is not None:
        sigma_psf_diag[2] = slice_thickness * fwhm_to_std / 1.2
    Sigma_psf = np.diag(sigma_psf_diag**2)
    L = affine3x3[:3, :3] / np.linalg.norm(affine3x3[:3, :3], axis=0)
    Sigma_psf = L @ Sigma_psf @ L.T
    return Sigma_psf


def init_optim(gsvr, cfg_gsvr, max_epochs):
    """
    Builds the AdamW optimiser with per-parameter-group learning rates and a
    cosine-annealing LR scheduler.  When coarse-to-fine is enabled the first
    cosine phase spans the epochs until the second growth step; subsequent
    phases are created by ``restart_scheduler`` at each growth step.  Without
    coarse-to-fine a single cosine spans the full training run.

    Each parameter group stores its ``initial_lr`` so warm restarts can
    restore peak learning rates after a growth step.

    Args:
        gsvr:       GaussianSVR model instance.
        cfg_gsvr:   GSVR config dict containing 'learning_rates',
                    'motion_correction', 'slice_scaling', and
                    'slice_weighting' keys.
        max_epochs: Total number of training epochs.

    Returns:
        optimizer:   torch.optim.AdamW instance.
        scheduler:   CosineAnnealingLR scheduler for the first phase.
        grad_scaler: torch.amp.GradScaler for mixed-precision training.
    """
    MC = cfg_gsvr['motion_correction']
    SS = cfg_gsvr['slice_scaling']
    lrs = cfg_gsvr['learning_rates']
    gprim = gsvr.gaussian_primitives
    param_groups = [
        {'params': gprim.mu, 'lr': lrs['g_primitives_mu'], 'name': 'gp_mu'},
        {'params': gprim.scaling, 'lr': lrs['g_primitives_scale'], 'name': 'gp_scaling' },
        {'params': gprim.rotation_g, 'lr': lrs['g_primitives_rot'], 'name': 'gp_rotation_g'},
        {'params': gprim.raw_color, 'lr': lrs['g_primitives_color'], 'name': 'gp_color'},
    ]

    if MC:
        param_groups.append({'params': gsvr.rotation_mc, 'lr': lrs['gsvr_mc_rot']})
        param_groups.append({'params': gsvr.translation_mc, 'lr': lrs['gsvr_mc_trans']})
    if SS:
        param_groups.append({'params': gsvr.slice_scales, 'lr': lrs['gsvr_slice_scale']})

    # Store initial LR for warm restarts
    for group in param_groups:
        group['initial_lr'] = group['lr']

    optimizer = torch.optim.AdamW(param_groups)

    # Compute T_max for the first cosine phase
    c2f_cfg = cfg_gsvr.get('coarse_to_fine', {})
    if c2f_cfg.get('enabled', False):
        schedule = sorted(c2f_cfg['growth_schedule'], key=lambda x: x[0])
        # First phase runs until the second growth step (first is epoch 0)
        T_max = schedule[1][0] if len(schedule) > 1 else max_epochs
    else:
        T_max = max_epochs

    eta_min_factor = cfg_gsvr.get('lr_eta_min_factor', 0.01)
    min_base_lr = min(g['initial_lr'] for g in optimizer.param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(T_max, 1), eta_min=eta_min_factor * min_base_lr)

    grad_scaler = torch.amp.GradScaler()
    return optimizer, scheduler, grad_scaler


def restart_scheduler(optimizer, T_max, eta_min_factor=0.01):
    """
    Warm-restart the LR schedule after a coarse-to-fine growth step.

    Resets every parameter group's learning rate back to its initial value
    and creates a fresh CosineAnnealingLR scheduler that will decay over
    ``T_max`` epochs.  This gives newly activated Gaussians a high initial
    LR to settle into their positions.

    Args:
        optimizer:       AdamW optimiser whose param groups carry ``initial_lr``.
        T_max:           Number of epochs in this phase (until next growth step
                         or end of training).
        eta_min_factor:  LR decays to this fraction of the smallest initial LR.

    Returns:
        scheduler: Fresh CosineAnnealingLR instance.
    """
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr']
    min_base_lr = min(g['initial_lr'] for g in optimizer.param_groups)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(T_max, 1), eta_min=eta_min_factor * min_base_lr)

def topK_neighbors(gpu_topk, g_primitives_mu, K, coords):
    """
    Finds the K nearest Gaussian primitives for each query coordinate using a
    FAISS GPU flat L2 index.  The index is rebuilt each call so that updated
    primitive positions (after gradient steps) are reflected.

    Args:
        gpu_topk:          faiss.GpuIndexFlatL2 instance.
        g_primitives_mu:   (M, 3) current Gaussian mean positions.
        K:                 Number of nearest neighbours to retrieve.
        coords:            (N, 3) query coordinates.

    Returns:
        top_k_idcs: (N, K) integer indices into g_primitives_mu.
    """
    gpu_topk.reset()
    gpu_topk.add(g_primitives_mu)
    _, top_k_idcs = gpu_topk.search(coords, K) # Search for nearest neighbors in the faiss index
    if (top_k_idcs >= g_primitives_mu.shape[0]).any() or (top_k_idcs < 0).any():
        print("Error: Faiss returned invalid indices!")
        top_k_idcs = torch.clamp(top_k_idcs, min=0, max=g_primitives_mu.shape[0]-1)
    return top_k_idcs
    
def compute_slice_outlier_weights(residuals, slice_ids, n_slices,
                                   k=4.0, mad_floor_abs=0.02):
    """
    Robust per-slice outlier weights from per-voxel residuals.

    Uses one-sided Welsch (soft) weighting:
        w = exp(-0.5 * (max(z, 0) / k)^2)
        z = (slice_mean_residual - global_median) / mad_eff

    Why Welsch instead of Tukey bisquare:
      - Welsch weights are strictly positive (smooth Gaussian decay), unlike
        Tukey's hard zero outside |z| < k. Hard-rejected slices stop receiving
        supervised gradient on their motion-correction parameters; those
        parameters then drift under regularisation, the slice mis-registers,
        its residual grows, and the next update confirms the rejection. On
        clean data this manifests as a cascade (e.g. 16 → 77/77 rejections
        in two updates). Soft weights keep every slice supervised enough to
        recover.

    Why one-sided (max(z, 0)):
      - "Outlier" here means high residual, not deviation from the median
        in either direction. A slice fit better than typical (z < 0) should
        not be down-weighted.

    Why an absolute MAD floor:
      - The previous relative floor (0.2 * med) shrinks with convergence:
        at low residual levels the threshold tightens and natural per-slice
        anatomy variance (some slices intersect more high-contrast tissue)
        looks like outlier evidence. An absolute floor in normalized
        intensity units (data ~[0, 1]) keeps the threshold meaningful.

    Why a permissive k=4:
      - The 95%-Gaussian-efficiency constant for Welsch is ~2.985. SVR
        residuals across slices are non-Gaussian (anatomy-driven content
        variance), so a more permissive threshold is appropriate. At z=3
        Welsch with k=4 gives w≈0.71 (current Tukey k=3 gives w=0).

    Caller is expected to apply a rejection-rate guard against catastrophic
    updates (see info['n_below_0.1']).

    Args:
        residuals:     (N,) per-voxel absolute residuals (detached).
        slice_ids:     (N,) integer slice index per voxel.
        n_slices:      Total number of slices across all stacks.
        k:             Welsch threshold in MAD units (default 4.0).
        mad_floor_abs: Absolute floor for the MAD scale, in normalized
                       intensity units (default 0.02 = 2% of [0, 1] range).

    Returns:
        weights: (S,) per-slice weights in [0, 1]; inactive slices = 0.
        active:  (S,) bool mask of slices with at least one voxel.
        info:    dict with diagnostic scalars: 'med', 'mad', 'mad_eff',
                 'z_max', 'n_below_0.1'.
    """
    slice_sum = torch.zeros(n_slices, device=residuals.device)
    slice_count = torch.zeros(n_slices, device=residuals.device)
    slice_sum.scatter_add_(0, slice_ids, residuals)
    slice_count.scatter_add_(0, slice_ids, torch.ones_like(residuals))
    active = slice_count > 0  # slices with at least one masked voxel
    slice_mean = slice_sum / slice_count.clamp(min=1)

    # Robust location/scale estimated only over active slices — empty
    # boundary slices (mean=0) would otherwise corrupt both estimators.
    active_means = slice_mean[active]
    med = active_means.median()
    mad = (active_means - med).abs().median() * 1.4826  # Gaussian consistency
    mad_eff = torch.clamp(mad, min=mad_floor_abs)

    z = (slice_mean - med) / mad_eff
    z_pos = z.clamp(min=0.0)  # only down-weight above-median (high-residual) slices
    weights = torch.exp(-0.5 * (z_pos / k) ** 2)
    weights = torch.where(active, weights, torch.zeros_like(weights))

    info = {
        'med': med.item(),
        'mad': mad.item(),
        'mad_eff': mad_eff.item(),
        'z_max': z[active].max().item() if active.any() else 0.0,
        'n_below_0.1': int(((weights < 0.1) & active).sum().item()),
    }
    return weights, active, info


def loss_composition(values_pred, values_gt, q, t, lambda_reg, scale_target, scale_primitives,
                     voxel_weights=None, num_active=None,
                     lambda_mc_rot=0.0, lambda_mc_trans=0.0,
                     color=None, g2g_knn_idcs=None, lambda_color_tv=0.0):
    """
    Computes the total training loss and its components.

    The optimised loss is:
        loss = loss_sr
             + lambda_reg      * mean(max(0, scale_target - log_std)^2)  # one-sided
             + lambda_mc_rot   * mean(q_imag^2)
             + lambda_mc_trans * mean(t^2)
             + lambda_color_tv * mean(|c_i - c_j|)        # graph-TV on colour

    The scale regulariser is a **one-sided lower bound**: only Gaussians whose
    log-scale falls below `scale_target` are penalised; larger Gaussians are
    free. This permits a heterogeneous scale distribution — wide background
    carriers for low-frequency / homogeneous regions plus sharper detail
    Gaussians on top — without anchoring every primitive to one size.

    Rotation and translation penalties are split because their magnitudes
    live on very different scales: a quaternion imaginary part of 0.1
    already encodes ~11° of rotation, while a translation of 0.1 mm is
    negligible.  Separate weights allow independent tuning.

    Graph-TV on colour: penalises L1 differences between each active Gaussian's
    colour and the colours of its K_color nearest Gaussian neighbours (via
    `g2g_knn_idcs`).  This couples otherwise-independent per-Gaussian colour
    parameters so that under-supervised Gaussians inherit smooth colour from
    their neighbours instead of drifting to spurious values — directly attacks
    the colour-domain origin of between-slice striping.  The K-NN graph is
    built only at FAISS rebuild cadence (every `topK_every` epochs) and reused
    between rebuilds; per-step cost here is one gather + L1.

    Args:
        values_pred:       (N,) predicted intensities.
        values_gt:         (N,) ground-truth intensities.
        q:                 (S, 4) per-slice rotation quaternions.
        t:                 (S, 3) per-slice translations.
        lambda_reg:        Scale regularisation weight (one-sided lower bound).
        scale_target:      Lower bound on Gaussian log-scale. Penalises only
                           scales below this value; larger Gaussians unconstrained.
        scale_primitives:  (K, 3) log-scale parameters of all Gaussian primitives.
        voxel_weights:     (N,) optional per-voxel weights from outlier handling.
                           Detached from the computation graph; derived from
                           per-slice residual-based outlier detection.
        num_active:        Number of active Gaussians (coarse-to-fine). When set,
                           scale regularisation is applied only to the first
                           num_active primitives. None = all (default).
        lambda_mc_rot:     Rotation regularisation weight. Penalises quaternion
                           imaginary parts (deviation from identity).
                           0 = no rotation penalty (default).
        lambda_mc_trans:   Translation regularisation weight. Penalises translation
                           magnitude. 0 = no translation penalty (default).
        color:             (K,) Gaussian colour values (post-softplus, the
                           intensities actually used in the forward). Required
                           when lambda_color_tv > 0.
        g2g_knn_idcs:      (num_active, K_color) int indices: for each active
                           Gaussian, its K_color nearest Gaussian neighbours
                           (self-neighbour excluded). Required when
                           lambda_color_tv > 0.
        lambda_color_tv:   Graph-TV weight on colour. 0 = no penalty (default).

    Returns:
        loss:      Scalar total loss.
        loss_dict: Dict with scalar items 'sr', 'sr_reg', 'mc_rot', 'mc_trans',
                   'color_tv'.
    """
    if voxel_weights is not None:
        loss_sr = (voxel_weights * (values_pred - values_gt).abs()).mean()
    else:
        loss_sr = F.l1_loss(values_pred, values_gt)
    loss_mc_rot = (q[:, 1:]**2).mean()
    loss_mc_trans = (t**2).mean()
    active_scales = scale_primitives[:num_active] if num_active is not None else scale_primitives
    # One-sided lower-bound hinge: zero gradient above scale_target; quadratic
    # ramp below. Lets large Gaussians cover homogeneous low-frequency regions
    # while preventing collapse into degenerate delta-like spikes.
    loss_sr_reg = lambda_reg * F.relu(scale_target - active_scales).pow(2).mean()

    if lambda_color_tv > 0.0 and color is not None and g2g_knn_idcs is not None:
        n_active = num_active if num_active is not None else color.shape[0]
        # (n_active, K_color) — colour differences between each active Gaussian
        # and its K_color spatial neighbours.
        diffs = color[:n_active].unsqueeze(1) - color[g2g_knn_idcs]
        loss_color_tv = lambda_color_tv * diffs.abs().mean()
    else:
        loss_color_tv = torch.zeros((), device=values_pred.device)

    loss = (loss_sr + loss_sr_reg
            + lambda_mc_rot * loss_mc_rot
            + lambda_mc_trans * loss_mc_trans
            + loss_color_tv)
    loss_dict = {
        'sr': loss_sr.item(),
        'sr_reg': loss_sr_reg.item(),
        'mc_rot': loss_mc_rot.item(),
        'mc_trans': loss_mc_trans.item(),
        'color_tv': loss_color_tv.item(),
    }
    return loss, loss_dict


def compute_adaptive_scale_target(base, num_active, max_gaussians, alpha):
    """
    Inflates the scale regularisation target when fewer Gaussians are active.

    With coarse-to-fine, early phases use fewer, larger Gaussians.  The target
    log-scale is increased so scale regularisation encourages the appropriate
    spatial extent instead of penalising the necessarily larger early primitives.

    Formula:  scale_target_eff = base + alpha * ln(max_gaussians / num_active)

    Args:
        base:           Baseline scale_target from config (e.g. 0.5).
        num_active:     Current number of active Gaussians.
        max_gaussians:  Total (pre-allocated) number of Gaussians.
        alpha:          Scaling coefficient (config: scale_target_alpha).

    Returns:
        Effective scale target (float).
    """
    return base + alpha * math.log(max_gaussians / max(num_active, 1))


def create_vis_grid(grid_shape, bbox_wrld, device):
    """
    Creates a dense 3D coordinate grid spanning the world-space bounding box.

    Args:
        grid_shape: (3,) desired number of voxels along each axis.
        bbox_wrld:  [min, max] bounding box, each (3,) in world coordinates.
        device:     Torch device.

    Returns:
        coords_flat: (N, 3) tensor of world-space coordinates, where
                     N = prod(grid_shape).
    """

    coords_axes = [torch.linspace(bbox_wrld[0][i], bbox_wrld[1][i], round(s)) for i, s in enumerate(grid_shape)] # TODO: check for correct size
    grid_x, grid_y, grid_z = torch.meshgrid(*coords_axes, indexing='ij')
    coords_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1)
    coords_flat = coords_flat.to(device)
    return coords_flat


def build_recon_mask(gsvr, coords, slice_centers, slice_idcs, bbox_world, spacing, shape_rec,
                     device, dilation_voxels=2, closing_voxels=5, vote_threshold=2):
    """
    Builds a reconstruction-grid brain mask in **aligned world space** via a
    per-recon-voxel **majority vote** across distinct contributing slices.

    Each supervisory brain voxel is motion-corrected into aligned world space
    and rasterised into the recon grid. For each recon voxel we count how many
    *distinct* slices contributed at least one voxel to it. The voxel is in
    the mask iff that count meets `vote_threshold`.

    Rationale: a misaligned slice scatters its voxels to a wrong world
    location where typically no other (correctly-aligned) slice's voxels land —
    so its wrong region gets exactly 1 vote and is filtered out by
    `vote_threshold ≥ 2`. True-brain recon voxels almost always receive votes
    from multiple slices/stacks and survive.

    Post-processing: (1) outward dilation by `dilation_voxels` via 3×3×3
    max-pool to grow the boundary; (2) symmetric morphological closing by
    `closing_voxels` (dilate-then-erode) to bridge open gaps narrower than
    `2 * closing_voxels` voxels without growing the outer boundary further —
    `binary_fill_holes` alone only catches topologically enclosed gaps;
    (3) `binary_fill_holes` for ventricles and other enclosed cavities;
    (4) keep the connected component closest to the recon-grid centre.

    Args:
        gsvr:           GaussianSVR model (provides `motion_correction`).
        coords:         (N, 3) world-space brain-voxel coords (un-corrected).
        slice_centers:  (N, 3) per-voxel slice-centre world coord (rotation pivot).
        slice_idcs:     (N,) per-voxel slice ID.
        bbox_world:     (2, 3) [min, max] world-space bounding box.
        spacing:        (3,) recon grid voxel spacing in mm.
        shape_rec:      (3,) recon grid shape.
        device:         Torch device.
        dilation_voxels:Outward growth at the boundary (3×3×3 max-pool
                        iterations). Each step adds one voxel of dilation
                        in every direction.
        closing_voxels: Symmetric dilate-then-erode iterations applied after
                        the outward dilation. Bridges open gaps narrower than
                        `2 * closing_voxels` voxels without growing the outer
                        boundary further. Set to 0 to disable.
        vote_threshold: Minimum number of *distinct slices* whose voxels must
                        land on a recon voxel for it to be in the mask
                        (default 2). Raise to 3+ if single mis-aligned-but-
                        agreeing pairs slip through.

    Returns:
        mask: (D, H, W) float32 binary tensor on `device`, 1.0 inside brain.
    """
    with torch.no_grad():
        coords_aligned, _ = gsvr.motion_correction(coords, slice_centers, slice_idcs, None)

        origin = bbox_world[0] if torch.is_tensor(bbox_world) else torch.as_tensor(bbox_world[0], device=device, dtype=coords_aligned.dtype)
        if not torch.is_tensor(origin):
            origin = torch.as_tensor(origin, device=device, dtype=coords_aligned.dtype)
        spacing_t = torch.as_tensor(spacing, device=device, dtype=coords_aligned.dtype)

        vox = ((coords_aligned - origin) / spacing_t).round().long()
        shape_t = torch.as_tensor(list(shape_rec), device=device, dtype=torch.long)
        inb = ((vox >= 0) & (vox < shape_t)).all(dim=1)
        vox = vox[inb]
        sids = slice_idcs[inb].long()

        D, H, W = int(shape_rec[0]), int(shape_rec[1]), int(shape_rec[2])
        n_recon = D * H * W
        flat_recon = vox[:, 0] * (H * W) + vox[:, 1] * W + vox[:, 2]   # (M,)

        # Count distinct (recon_voxel, slice) pairs by encoding both into a
        # single long key, taking uniques, and scattering one count per
        # unique pair into the recon voxel's vote bin. This naturally
        # de-duplicates intra-slice repeats so each slice contributes at
        # most one vote per recon voxel.
        n_slices = int(sids.max().item()) + 1
        key = flat_recon * n_slices + sids
        unique_keys = torch.unique(key)
        recon_idx = (unique_keys // n_slices).long()
        vote_count = torch.zeros(n_recon, device=device, dtype=torch.int32)
        vote_count.scatter_add_(0, recon_idx,
                                torch.ones_like(recon_idx, dtype=torch.int32))

        mask = (vote_count >= vote_threshold).reshape(D, H, W).to(torch.float32)

        if dilation_voxels > 0:
            m = mask.unsqueeze(0).unsqueeze(0)
            for _ in range(dilation_voxels):
                m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
            mask = m.squeeze(0).squeeze(0)

        # CPU round-trip for morphological closing + connected-component
        # selection. Cheap (~100 ms on ~8 M-voxel grids) and happens once
        # at render time.
        mask_np = mask.cpu().numpy().astype(bool)

        # Symmetric closing: bridges open gaps (channels to background
        # wider than `dilation_voxels`) that binary_fill_holes can't catch
        # because they are not topologically enclosed. Dilation + matched
        # erosion cancel on smooth surfaces, so the outer boundary is
        # preserved while gaps narrower than 2*closing_voxels are sealed.
        # 3×3×3 cube structuring element matches the GPU max-pool above.
        if closing_voxels > 0:
            structure = scipy.ndimage.generate_binary_structure(3, 3)
            mask_np = scipy.ndimage.binary_closing(
                mask_np, structure=structure, iterations=closing_voxels)

        # Fill internal cavities (ventricles, mask-coverage gaps) so the brain
        # becomes a single solid region before component analysis.
        mask_np = scipy.ndimage.binary_fill_holes(mask_np)

        # Keep the connected component whose centroid is closest to the recon
        # grid centre. The brain sits near the bbox centre by construction
        # (the bbox is derived from the union of brain coords); misaligned-
        # slice scatter typically lands off-centre and forms separate CCs.
        labeled, n_cc = scipy.ndimage.label(mask_np)
        if n_cc == 0:
            print("  Mask: WARNING — empty after fill_holes; returning all-zero mask")
        elif n_cc == 1:
            pass  # nothing to filter
        else:
            grid_centre = np.array([D / 2.0, H / 2.0, W / 2.0])
            best_label, best_dist = 0, float('inf')
            for lbl in range(1, n_cc + 1):
                com = np.array(scipy.ndimage.center_of_mass(mask_np, labeled, lbl))
                dist = float(np.linalg.norm(com - grid_centre))
                if dist < best_dist:
                    best_dist = dist
                    best_label = lbl
            mask_np = (labeled == best_label)
            print(f"  Mask: kept central CC (label {best_label}, "
                  f"centroid-to-centre={best_dist:.1f} vox), discarded {n_cc - 1} other CCs")

        mask = torch.from_numpy(mask_np.astype(np.float32)).to(device)

        n_in_mask = int(mask.sum().item())
        n_max_votes = int(vote_count.max().item())
        print(f"  Mask: vote_threshold={vote_threshold}, "
              f"in-mask voxels={n_in_mask}, max votes/recon-voxel={n_max_votes}")
    return mask


def visualize_gaussians(gsvr, gpu_topk, K, vis_grid_flat, shape_rec, cfg_data, output_file_path, bbox_world,
                        min_value_stack, max_value_stack,
                        coords_nrmd, slice_centers_nrmd, slice_idcs,
                        ttime=0, device='cuda'):
    """
    Renders the full 3D Gaussian field onto a dense grid, applies a motion-
    corrected brain mask, and saves the result as a NIfTI volume.

    Inference uses an isotropic PSF derived from the reconstruction spacing.
    The volume is evaluated in chunks of 1M points to stay within GPU memory.
    Predicted intensities are de-normalised back to the original input range
    before saving, so the output histogram matches the input stacks.

    The output mask is built by `build_recon_mask` from the (motion-corrected)
    brain supervisory coords — so it tracks the learned per-slice transforms.

    Args:
        gsvr:                GaussianSVR model (may be torch.compile-wrapped).
        gpu_topk:            faiss.GpuIndexFlatL2 instance.
        K:                   Number of nearest neighbours.
        vis_grid_flat:       (N, 3) dense world-space coordinate grid.
        shape_rec:           (3,) target reconstruction shape in voxels.
        cfg_data:            Data config dict; uses 'reconstruction.spacing'.
        output_file_path:    Directory to save the reconstruction NIfTI.
        bbox_world:          [min, max] world-space bounding box, each (3,);
                             sets the NIfTI origin so the output is physically
                             aligned to the input stacks.
        min_value_stack:     Minimum intensity of the input data before normalisation.
        max_value_stack:     Maximum intensity of the input data before normalisation.
        coords_nrmd:         (N, 3) supervisory brain-voxel world coords (pre-MC).
        slice_centers_nrmd:  (N, 3) per-voxel slice centre (rotation pivot).
        slice_idcs:          (N,) per-voxel slice ID.
        ttime:               Accumulated training time in seconds (used in filename).
        device:              Torch device.
    """
    batch_size = 1_000_000
    spacing_rec = cfg_data['reconstruction']['spacing']
    covs_psf_inference = torch.from_numpy(generate_cov_psf(np.eye(3), spacing_rec, slice_thickness=None)).to(device)
    values_pred_flat = torch.empty((vis_grid_flat.shape[0],), device=device)
    for step in range(0, vis_grid_flat.shape[0], batch_size):
        with torch.no_grad():
            coords = vis_grid_flat[step:step+batch_size]
            covs_pdf_inf_expanded = covs_psf_inference.unsqueeze(0).expand(coords.shape[0], -1, -1)
            num_active = gsvr.gaussian_primitives.num_active.item()
            topK_idcs = topK_neighbors(gpu_topk, gsvr.gaussian_primitives.mu.data[:num_active], K, coords)
            values_pred_flat[step:step+batch_size] = gsvr(coords, topK_idcs, covs_psf=covs_pdf_inf_expanded)

    apply_mask = cfg_data['reconstruction'].get('apply_mask', True)
    if apply_mask:
        rec_cfg = cfg_data['reconstruction']
        recon_mask = build_recon_mask(
            gsvr, coords_nrmd, slice_centers_nrmd, slice_idcs,
            bbox_world, spacing_rec, shape_rec, device,
            dilation_voxels=rec_cfg.get('mask_dilation_voxels', 3),
            closing_voxels=rec_cfg.get('mask_closing_voxels', 5),
            vote_threshold=rec_cfg.get('mask_vote_threshold', 2))
        values_pred_flat = values_pred_flat * recon_mask.reshape(-1)

    values_pred = values_pred_flat \
        .reshape(np.array(shape_rec, dtype=int).tolist()).detach().cpu().numpy()
    values_pred = (values_pred * (max_value_stack - min_value_stack) + min_value_stack).clip(0.0, None)
    origin = np.array(bbox_world[0].cpu()) if hasattr(bbox_world[0], 'cpu') else np.array(bbox_world[0])
    rec_affine = np.diag(list(spacing_rec) + [1.0])
    rec_affine[:3, 3] = origin
    rec_nii = nib.Nifti1Image(values_pred, rec_affine)
    filename_pred = os.path.join(output_file_path, f'reconstruction_ttime={ttime:.2f}s.nii.gz')
    nib.save(rec_nii, filename_pred)


def preprocess_image(input_spec, cfg_data):
    """
    Load a single NIfTI image (stack or motion-corrected slice) and apply
    optional ANTs N4 / denoise preprocessing.

    The image is multiplied by its brain mask before preprocessing so that ANTs
    operates only on in-mask voxels. MRI intensities are clamped to ≥ 0 on the
    way out.

    Args:
        input_spec: dict from `data_inputs.resolve_subject_inputs()` with keys
                    `image_path`, `mask_path` (or None), and `label`.
        cfg_data:   full data config dict (for `preprocessing` toggles).

    Returns:
        img_data: (H, W, D) float ndarray, zeroed outside the brain mask.
        mask:     (H, W, D) int32 brain mask matching `img_data` in shape.
        affine:   (4, 4) world-space affine matrix.
        spacing:  (3,) voxel spacing in mm.
    """
    img = nib.load(input_spec['image_path'])
    img_data = img.get_fdata()
    affine = img.affine
    spacing = img.header.get_zooms()[:3]

    if input_spec['mask_path'] is None:
        # No mask provided — use non-zero voxels as the brain.
        mask = (img_data > 0.0).astype(np.int32)
    else:
        # Reshape to match image shape so e.g. (H, W, D, 1) masks load safely
        # and (H, W, 1) slice masks aren't accidentally squeezed to (H, W).
        mask_data = nib.load(input_spec['mask_path']).get_fdata()
        mask = (mask_data.reshape(img_data.shape) > 0).astype(np.int32)

    img_data = img_data * mask

    if cfg_data['preprocessing']['bias_field_correction']:
        print(f"Applying bias field correction to {input_spec['label']}...")
        img_data = ants.n4_bias_field_correction(
            ants.from_nibabel(nib.Nifti1Image(img_data, affine)))
        if not cfg_data['preprocessing']['denoise']:
            img_data = img_data.to_nibabel().get_fdata()
    if cfg_data['preprocessing']['denoise']:
        if not cfg_data['preprocessing']['bias_field_correction']:
            img_data = ants.from_nibabel(nib.Nifti1Image(img_data, affine))
        print(f"Denoising {input_spec['label']}...")
        img_data = ants.denoise_image(img_data).to_nibabel().get_fdata()
    img_data = img_data.clip(0.0, None)  # MRI intensities must be non-negative
    return img_data, mask, affine, spacing


def load_data(cfg_data):
    """
    Load and assemble per-voxel training tensors from a data config dict.

    Dispatches to either stacks mode (`subject.input_stacks`) or slices mode
    (`subject.slices_dir`) via `data_inputs.resolve_subject_inputs()`. In both
    modes the same per-input assembly logic produces a flat batch of
    `(world_coord, intensity, slice_id)` tuples plus a per-slice PSF covariance.

    In slices mode each input file is one slice (z-extent 1) with its own
    affine, so `n_slices_global` equals the number of slice files.
    """
    from data_inputs import resolve_subject_inputs

    inputs = resolve_subject_inputs(cfg_data)

    all_values = []
    all_coords = []
    all_affines = []
    all_stack_imgs = []
    all_cov_psf = []
    all_slice_centers = []  # one per voxel; slice = thick-slice (z-axis of input)
    all_slice_idcs = []
    n_slices_global = 0

    for input_spec in inputs:
        img_data, mask_brain, affine, spacing = preprocess_image(input_spec, cfg_data)
        # Brain-only supervision: select voxels strictly inside the brain mask.
        # The reconstruction is masked at inference time via the motion-corrected
        # brain coords (see build_recon_mask); no halo voxels needed in training.
        coords_vxl = np.argwhere(mask_brain > 0)
        coords_vxl_homo = np.hstack([coords_vxl, np.ones((len(coords_vxl), 1))])
        coords_wrld = (affine @ coords_vxl_homo.T).T[:, :3]

        # get slice centers in world space
        nx, ny, nz = img_data.shape[:3]
        slice_axis_centers = np.array([nx / 2.0, ny / 2.0])
        slice_centers_voxel = np.array([np.hstack([slice_axis_centers, k, 1]) for k in range(nz)])
        slice_centers_world = (affine @ slice_centers_voxel.T).T[:, :3]
        slice_indices = coords_vxl[:, 2].astype(int)
        slice_centers_wrld = slice_centers_world[slice_indices]
        slice_idcs_global = slice_indices + n_slices_global
        n_slices_global += nz
        cov_psf = generate_cov_psf(affine, spacing, input_spec['slice_thickness'])
        cov_psf = np.broadcast_to(cov_psf, (nz, 3, 3))

        # add to global lists
        all_stack_imgs.append(img_data)
        all_coords.append(coords_wrld)
        all_values.append(img_data[coords_vxl[:, 0], coords_vxl[:, 1], coords_vxl[:, 2]])
        all_cov_psf.append(cov_psf)
        all_slice_centers.append(slice_centers_wrld)
        all_slice_idcs.append(slice_idcs_global)
        all_affines.append(affine)

    all_values = np.concatenate(all_values, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)
    all_slice_centers = np.concatenate(all_slice_centers, axis=0)
    all_slice_idcs = np.concatenate(all_slice_idcs, axis=0)
    all_cov_psf = np.concatenate(all_cov_psf, axis=0)
    all_affines = np.stack(all_affines, axis=0)

    bbox_min_raw = np.min(all_coords, axis=0)
    bbox_max_raw = np.max(all_coords, axis=0)
    padding = (bbox_max_raw - bbox_min_raw) * 0.05  # 5% of extent on each side, symmetric
    bbox_min = bbox_min_raw - padding
    bbox_max = bbox_max_raw + padding
    bbox_world = [bbox_min, bbox_max]
    bbox_world_size = bbox_max - bbox_min
    all_coords_nrmd = all_coords
    all_slice_centers_nrmd = all_slice_centers

    min_value_stack = all_values.min()  # >= 0 after clip in preprocess_image
    max_value_stack = all_values.max()
    all_values_nrmd = (all_values - min_value_stack) / (max_value_stack - min_value_stack)
    all_stack_imgs_nrmd = [(img.clip(0.0, None) - min_value_stack) / (max_value_stack - min_value_stack) for img in all_stack_imgs]

    shape_reconstruction = [round(b / s) + 1 for b, s in zip(bbox_world_size, cfg_data['reconstruction']['spacing'])]
    return [all_stack_imgs_nrmd, all_affines,
            all_slice_idcs, all_slice_centers_nrmd,
            all_coords_nrmd, all_values_nrmd, all_cov_psf,
            bbox_world, bbox_world_size, shape_reconstruction, n_slices_global,
            min_value_stack, max_value_stack]


