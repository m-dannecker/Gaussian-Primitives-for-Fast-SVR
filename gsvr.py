"""
Gaussian Slice-to-Volume Reconstruction (GSVR) model.

Represents a 3D MRI volume as a set of anisotropic 3D Gaussian primitives and
optimises them to fit a set of thick 2D MRI slice stacks. Supports per-slice
rigid motion correction and optional per-slice outlier scaling.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Suppress the specific internal warning from torch.compile
warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")
torch.set_float32_matmul_precision('high')

def build_rotation_matrix_from_quaternion(q):
    '''
    Builds a batch of 3x3 rotation matrices from unit quaternions.

    Args:
        q: (K, 4) tensor, quaternion components ordered (w, x, y, z).
           Inputs are L2-normalised internally before conversion.

    Returns:
        R: (K, 3, 3) rotation matrix tensor.
    '''
    # Normalize quaternions to ensure they are unit quaternions
    q_norm = torch.nn.functional.normalize(q, p=2, dim=1)
    
    w, x, y, z = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]
    
    K = q.shape[0]
    R = torch.empty((K, 3, 3), device=q.device, dtype=q.dtype)

    # Pre-compute reused terms
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Fill the rotation matrix
    R[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)

    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
    R[:, 1, 2] = 2.0 * (yz - wx)

    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)
    
    return R


@torch.compile
def fused_mahalanobis_distance(sigmas_nk, x_minus_mu_nk):
    """
    Computes the squared Mahalanobis distance v^T Sigma^{-1} v for each
    (point, Gaussian) pair. The 3x3 matrix inverse and the quadratic form are
    fused into a single compiled kernel using Cramer's rule, avoiding explicit
    materialisation of the inverse tensor.

    Args:
        sigmas_nk:      (N, K, 3, 3) covariance matrices (Gaussian + PSF).
        x_minus_mu_nk:  (N, K, 3)   centred coordinates (x - mu).

    Returns:
        result: (N, K) squared Mahalanobis distances.
    """
    # 1. Unpack Covariance Matrix (N, K, 3, 3)
    # This avoids slicing overhead in Python
    a = sigmas_nk[..., 0, 0]
    b = sigmas_nk[..., 0, 1]
    c = sigmas_nk[..., 0, 2]
    # d = sigmas_nk[..., 1, 0] # Sym: d=b
    e = sigmas_nk[..., 1, 1]
    f = sigmas_nk[..., 1, 2]
    # g = sigmas_nk[..., 2, 0] # Sym: g=c
    h = sigmas_nk[..., 2, 1] # Sym: h=f
    i = sigmas_nk[..., 2, 2]

    # 2. Compute Determinant (Sarrus Rule)
    # det = a(ei - fh) - b(di - fg) + c(dh - eg)
    # utilizing symmetry where d=b, g=c, h=f
    det = a * (e * i - f * h) - b * (b * i - f * c) + c * (b * h - e * c)
    inv_det = 1.0 / (det + 1e-8) # Epsilon for stability

    # 3. Compute Inverse Elements (Cramer's Rule / Adjugate)
    # Only need Upper Triangle due to symmetry in the final Quadratic Form? 
    # Actually, we need full mult or optimized quadratic form. 
    # Let's do explicit inverse elements.
    
    # inv_00 = (ei - fh) * inv_det
    # inv_01 = (ch - bi) * inv_det
    # inv_02 = (bf - ce) * inv_det
    # inv_11 = (ai - cg) * inv_det -> (ai - c^2)
    # inv_12 = (cd - af) * inv_det -> (cb - af)
    # inv_22 = (ae - bd) * inv_det -> (ae - b^2)
    
    inv_00 = (e * i - f * h) * inv_det
    inv_01 = (c * h - b * i) * inv_det
    inv_02 = (b * f - c * e) * inv_det
    inv_10 = inv_01 # Symmetric
    inv_11 = (a * i - c * c) * inv_det
    inv_12 = (c * b - a * f) * inv_det
    inv_20 = inv_02 # Symmetric
    inv_21 = inv_12 # Symmetric
    inv_22 = (a * e - b * b) * inv_det

    # 4. Compute v^T * Sigma^-1 * v
    # v = [x, y, z]
    vx = x_minus_mu_nk[..., 0]
    vy = x_minus_mu_nk[..., 1]
    vz = x_minus_mu_nk[..., 2]

    # Manual Matrix-Vector Multiplication to avoid creating a tensor for Sigma_inv
    # Row 0 dot v
    t0 = inv_00 * vx + inv_01 * vy + inv_02 * vz
    # Row 1 dot v
    t1 = inv_10 * vx + inv_11 * vy + inv_12 * vz
    # Row 2 dot v
    t2 = inv_20 * vx + inv_21 * vy + inv_22 * vz

    # Final dot product
    result = vx * t0 + vy * t1 + vz * t2
    
    return result 

@torch.compile
def fused_motion_correction_kernel(
    coords_n_3,
    quats_k_4,
    trans_k_3,
    slice_ids_n,
    slice_centers_n_3,
    sigma_psf_n_3_3=None
):
    """
    Applies per-slice rigid motion correction to coordinates and optionally
    to PSF covariance matrices.  All operations (quaternion->matrix conversion,
    coordinate transform, covariance rotation) are fused into one compiled
    kernel to minimise memory traffic.

    The transform is: x' = R_s (x - c_s) + t_s + c_s
    where s = slice_ids_n[n], c_s is the slice centre, R_s the rotation
    derived from quats_k_4[s], and t_s the translation.

    The PSF covariance is rotated as: Sigma' = R_s Sigma R_s^T.

    Args:
        coords_n_3:      (N, 3)    input world-space coordinates.
        quats_k_4:       (S, 4)    per-slice quaternions (w, x, y, z).
        trans_k_3:       (S, 3)    per-slice translations.
        slice_ids_n:     (N,)      integer slice index for each point.
        slice_centers_n_3: (N, 3) rotation centre for each point's slice.
        sigma_psf_n_3_3: (N, 3, 3) optional per-point PSF covariance matrices.

    Returns:
        coords_mc:  (N, 3)    motion-corrected coordinates.
        sigma_tf:   (N, 3, 3) rotated PSF covariances, or None.
    """
    # 1. Gather Motion Parameters for each point N
    # We must index inside the kernel to allow fusion of the gather + math
    # quats_n: (N, 4)
    q_w = quats_k_4[:, 0]
    q_x = quats_k_4[:, 1]
    q_y = quats_k_4[:, 2]
    q_z = quats_k_4[:, 3]
    
    t_x = trans_k_3[slice_ids_n, 0]
    t_y = trans_k_3[slice_ids_n, 1]
    t_z = trans_k_3[slice_ids_n, 2]

    # 2. Normalize Quaternion (in registers)
    inv_norm = torch.rsqrt(q_w*q_w + q_x*q_x + q_y*q_y + q_z*q_z + 1e-8)
    w = q_w * inv_norm
    x = q_x * inv_norm
    y = q_y * inv_norm
    z = q_z * inv_norm

    # 3. Construct Rotation Matrix Elements (in registers)
    x2 = x * x; y2 = y * y; z2 = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    r00 = 1.0 - 2.0 * (y2 + z2)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (x2 + z2)
    r12 = 2.0 * (yz - wx)
    
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (x2 + y2)

    # 4. Apply Coordinate Transform
    # coords = (R @ (coords - center)) + t + center
    
    # Shift to center
    c_x = coords_n_3[:, 0] - slice_centers_n_3[:, 0]
    c_y = coords_n_3[:, 1] - slice_centers_n_3[:, 1]
    c_z = coords_n_3[:, 2] - slice_centers_n_3[:, 2]

    # Rotate
    rot_x = r00[slice_ids_n] * c_x + r01[slice_ids_n] * c_y + r02[slice_ids_n] * c_z
    rot_y = r10[slice_ids_n] * c_x + r11[slice_ids_n] * c_y + r12[slice_ids_n] * c_z
    rot_z = r20[slice_ids_n] * c_x + r21[slice_ids_n] * c_y + r22[slice_ids_n] * c_z

    # Translate and Shift back
    final_x = rot_x + t_x + slice_centers_n_3[:, 0]
    final_y = rot_y + t_y + slice_centers_n_3[:, 1]
    final_z = rot_z + t_z + slice_centers_n_3[:, 2]
    
    coords_mc = torch.stack([final_x, final_y, final_z], dim=-1)

    # 5. Apply Sigma PSF Transform (if needed)
    # Sigma_new = R @ Sigma @ R.T
    sigma_tf = None
    if sigma_psf_n_3_3 is not None:
        s00 = sigma_psf_n_3_3[:, 0, 0]; s01 = sigma_psf_n_3_3[:, 0, 1]; s02 = sigma_psf_n_3_3[:, 0, 2]
        s10 = sigma_psf_n_3_3[:, 1, 0]; s11 = sigma_psf_n_3_3[:, 1, 1]; s12 = sigma_psf_n_3_3[:, 1, 2]
        s20 = sigma_psf_n_3_3[:, 2, 0]; s21 = sigma_psf_n_3_3[:, 2, 1]; s22 = sigma_psf_n_3_3[:, 2, 2]

        # First Matmul: Temp = R @ Sigma
        # We manually unroll to keep it in registers
        t00 = r00*s00 + r01*s10 + r02*s20
        t01 = r00*s01 + r01*s11 + r02*s21
        t02 = r00*s02 + r01*s12 + r02*s22
        
        t10 = r10*s00 + r11*s10 + r12*s20
        t11 = r10*s01 + r11*s11 + r12*s21
        t12 = r10*s02 + r11*s12 + r12*s22
        
        t20 = r20*s00 + r21*s10 + r22*s20
        t21 = r20*s01 + r21*s11 + r22*s21
        t22 = r20*s02 + r21*s12 + r22*s22

        # Second Matmul: Result = Temp @ R.T
        # Note: R.T means we dot with rows of R again
        # Res00 = row0(Temp) . row0(R)
        res00 = t00*r00 + t01*r01 + t02*r02
        res01 = t00*r10 + t01*r11 + t02*r12
        res02 = t00*r20 + t01*r21 + t02*r22
        
        res10 = t10*r00 + t11*r01 + t12*r02
        res11 = t10*r10 + t11*r11 + t12*r12
        res12 = t10*r20 + t11*r21 + t12*r22
        
        res20 = t20*r00 + t21*r01 + t22*r02
        res21 = t20*r10 + t21*r11 + t22*r12
        res22 = t20*r20 + t21*r21 + t22*r22

        sigma_tf = torch.stack([
            torch.stack([res00, res01, res02], dim=-1),
            torch.stack([res10, res11, res12], dim=-1),
            torch.stack([res20, res21, res22], dim=-1)
        ], dim=1)
        sigma_tf = sigma_tf[slice_ids_n]

    return coords_mc, sigma_tf


class GaussianSVR(nn.Module):
    """
    Top-level Gaussian Slice-to-Volume Reconstruction model.

    Wraps GaussianPrimitives with optional per-slice rigid motion correction
    and outlier handling.  Outlier handling has two components:

    1. **Per-slice intensity scales** (learnable) — model physical inter-slice
       intensity variation (B1 field, excitation profile differences).
       Parameterised via softplus and mean-normalised so the average scale is 1.
    2. **Per-slice outlier weights** (non-learnable buffer) — derived from
       per-voxel residuals using Tukey's bisquare function with MAD-based
       standardisation.  Down-weights slices corrupted by intra-slice motion
       or signal dropout without allowing the optimiser to game the weights.

    The forward pass evaluates the Gaussian field at query coordinates; motion
    correction is applied separately before the forward call so that its
    gradients flow through both the corrected coordinates and the Gaussian
    parameters.

    Args:
        D:               Spatial dimensionality (always 3).
        stack_imgs_nrmd: List of normalised numpy stacks, used for
                         content-adaptive initialisation.
        affines:         (S, 4, 4) array of NIfTI affine matrices.
        bbox_world:      [min, max] world-space bounding box (each (3,)).
        num_slices:      Total number of thick slices across all stacks.
        cfg_gsvr:        GSVR config dict (from YAML).
        device:          Torch device string.
    """

    def __init__(self, D, stack_imgs_nrmd, affines, bbox_world, num_slices, cfg_gsvr, device='cuda'):
        super().__init__()
        self.device = device
        self.D = 3
        self.mc = cfg_gsvr['motion_correction']
        self.slice_scaling = cfg_gsvr['slice_scaling']
        self.slice_weighting = cfg_gsvr['slice_weighting']
        self.slice_idcs = torch.arange(num_slices, device=device)
        scale_init = cfg_gsvr['g_primitives']['scale_init']
        self.gaussian_primitives = GaussianPrimitives(
            D, cfg_gsvr['g_primitives']['num_gaussians'],
            scale_init=scale_init,
        ).to(device)

        # Coarse-to-fine: optionally initialise only the first batch of Gaussians
        c2f_cfg = cfg_gsvr.get('coarse_to_fine', {})
        c2f_enabled = c2f_cfg.get('enabled', False)
        if c2f_enabled:
            initial_active = c2f_cfg['growth_schedule'][0][1]
            num_gaussians = cfg_gsvr['g_primitives']['num_gaussians']
            alpha = c2f_cfg.get('scale_target_alpha', 0.3)
            self.gaussian_primitives.num_active.fill_(initial_active)
            num_to_init = initial_active
            # Scale init should match the adaptive scale target for this phase,
            # so regularisation starts at its natural operating point rather
            # than immediately pushing scales upward from below.
            adapted_scale_init = scale_init + alpha * math.log(num_gaussians / max(initial_active, 1))
            with torch.no_grad():
                self.gaussian_primitives.scaling.data[:initial_active].fill_(adapted_scale_init)
        else:
            num_to_init = None  # init all

        if cfg_gsvr['g_primitives']['init_type'] == 'content_adaptive':
            self.gaussian_primitives.initialize_parameters_from_image(
                stack_imgs_nrmd, affines, bbox_world, lambda_init=0.0,
                device=self.device, num_to_init=num_to_init)

        # Motion Correction Parameters
        if self.mc:
            self.rotation_mc = nn.Parameter(torch.zeros(num_slices, 4))
            self.rotation_mc.data[:, 0] = 1.0
            self.translation_mc = nn.Parameter(torch.zeros(num_slices, 3))
        else:
            self.rotation_mc = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
            self.translation_mc = torch.tensor([[0.0, 0.0, 0.0]], device=device)

        # Outlier Handling
        if self.slice_scaling:
            # Per-slice intensity scale (learnable): softplus(0) ≈ 0.69,
            # mean-normalised to 1.0 during training
            self.slice_scales = nn.Parameter(torch.zeros(num_slices))
        if self.slice_weighting:
            # Per-slice outlier weight (non-learnable): updated from residuals
            self.register_buffer('slice_outlier_weights', torch.ones(num_slices))
        

    def forward(self, coords, top_k_idcs, covs_psf):
        """
        Evaluates the Gaussian field at the given (already motion-corrected)
        coordinates.

        Args:
            coords:      (N, 3) query coordinates in world space.
            top_k_idcs:  (N, K) FAISS k-NN indices into the Gaussian primitives.
            covs_psf:    (N, 3, 3) per-point PSF covariance matrices, or None.

        Returns:
            color_pred: (N,) predicted intensities.
        """
        return self.gaussian_primitives(coords, top_k_idcs, covs_psf)

    def motion_correction(self, coords, slice_centers, slice_ids, cov_psf=None):
        """
        Applies learnable per-slice rigid motion correction to coordinates and
        optionally rotates the PSF covariance into the corrected frame.

        Args:
            coords:        (N, 3) input world-space coordinates.
            slice_centers: (N, 3) rotation centre for each point's slice.
            slice_ids:     (N,)   integer slice index for each point.
            cov_psf:       (S, 3, 3) per-slice PSF covariances, or None.

        Returns:
            coords_mc:   (N, 3)    motion-corrected coordinates.
            cov_psf_tf:  (N, 3, 3) rotated PSF covariances, or None.
        """
        if self.mc:
            quats = self.rotation_mc
            trans = self.translation_mc
            # Call the fused kernel
            coords_mc, cov_psf_tf = fused_motion_correction_kernel(
                coords, quats, trans, slice_ids, slice_centers, cov_psf
            )
        else:
            coords_mc = coords
            cov_psf_tf = cov_psf[slice_ids] if cov_psf is not None else None
        return coords_mc, cov_psf_tf


class GaussianPrimitives(nn.Module):
    """
    A set of K anisotropic 3D Gaussian primitives representing a continuous
    intensity field.

    Each Gaussian k is parameterised by:
        mu         (3,)  — world-space mean position
        scaling    (3,)  — log-std along each principal axis; std = exp(scaling) mm,
                           variance = exp(2*scaling).  Clamped to [-3, 3] in forward.
        rotation_g (4,)  — orientation as a quaternion (w, x, y, z)
        raw_color  ()    — unconstrained intensity parameter; exposed as the read-only
                           `color` property via softplus(raw_color) ≥ 0

    The covariance is built as Sigma = R S R^T, where R is the rotation matrix
    and S = diag(exp(scaling)).  Evaluation uses a soft-normalised weighted sum
    over the K nearest primitives (found via FAISS), with the PSF covariance
    added to each Gaussian's covariance before computing the Mahalanobis distance.
    """

    def __init__(self, D, num_gaussians, scale_init=0.5):
        super().__init__()
        self.D = D
        self.num_gaussians = num_gaussians  # K

        # Mean: randomly initialised in [-1, 1] (overwritten by content-adaptive init)
        self.mu = nn.Parameter(torch.rand(num_gaussians, self.D) * 2 - 1)
        # Log-std: exp(scaling) = std in mm; covariance diagonal = exp(2*scaling)
        # Initialised to scale_init so Gaussians start at their target size.
        self.scaling = nn.Parameter(torch.ones(num_gaussians, self.D) * scale_init)
        # Rotation quaternion (w, x, y, z): initialised to identity
        self.rotation_g = nn.Parameter(torch.zeros(num_gaussians, 4))
        self.rotation_g.data[:, 0] = 1.0
        # Unconstrained raw intensity — exposed as softplus(raw_color) ≥ 0 via property
        self.raw_color = nn.Parameter(torch.zeros(num_gaussians))

        # Coarse-to-fine: number of currently active Gaussians.
        # Defaults to num_gaussians (all active) for backward compatibility.
        # When c2f is enabled the caller sets this to the initial active count.
        self.register_buffer('num_active', torch.tensor(num_gaussians, dtype=torch.long))

    @property
    def color(self):
        """Scalar intensity per Gaussian, guaranteed ≥ 0 via softplus activation."""
        return F.softplus(self.raw_color)

    @torch.no_grad()
    def split_and_activate(self, new_num_active, error_per_gaussian=None):
        """
        Grow the active set by splitting existing Gaussians.

        Selects the top candidates among currently active Gaussians ranked by
        (volume * error) — or volume alone if no error signal is available —
        and splits each parent into itself plus one child.

        The child is placed along the parent's principal axis of largest
        variance, and both parent and child have their scales reduced so the
        pair roughly covers the same region with finer detail.

        Args:
            new_num_active:     Target number of active Gaussians after growth.
            error_per_gaussian: (num_active,) mean absolute reconstruction error
                                per Gaussian, or None to rank by volume only.
        """
        old_n = self.num_active.item()
        num_to_add = new_num_active - old_n
        if num_to_add <= 0:
            return

        # --- Score active Gaussians: volume proxy * error ---
        scales_active = self.scaling.data[:old_n]
        volume = torch.exp(scales_active).prod(dim=1)  # (old_n,)
        if error_per_gaussian is not None and error_per_gaussian.shape[0] >= old_n:
            score = volume * error_per_gaussian[:old_n]
        else:
            score = volume

        # Allow a Gaussian to be split multiple times if num_to_add > old_n
        num_candidates = min(num_to_add, old_n)
        _, parent_idcs = score.topk(num_candidates)

        # If we need more splits than unique parents, repeat the top candidates
        if num_to_add > num_candidates:
            repeats = (num_to_add + num_candidates - 1) // num_candidates
            parent_idcs = parent_idcs.repeat(repeats)[:num_to_add]

        # --- Compute child positions along the parent's largest-variance axis ---
        R = build_rotation_matrix_from_quaternion(self.rotation_g.data[parent_idcs])  # (M, 3, 3)
        stds = torch.exp(self.scaling.data[parent_idcs])  # (M, 3)
        max_axis = stds.argmax(dim=1)  # (M,)
        # Column of R corresponding to the largest-std axis
        direction = R[torch.arange(num_to_add, device=R.device), :, max_axis]  # (M, 3)
        offset = 0.5 * stds[torch.arange(num_to_add, device=stds.device), max_axis].unsqueeze(1) * direction

        # --- Write child parameters into inactive slots ---
        child_slice = slice(old_n, old_n + num_to_add)
        self.mu.data[child_slice] = self.mu.data[parent_idcs] + offset
        self.rotation_g.data[child_slice] = self.rotation_g.data[parent_idcs]
        self.raw_color.data[child_slice] = self.raw_color.data[parent_idcs]

        # Reduce scales: log(std) -= log(sqrt(2)), i.e. std /= sqrt(2),
        # so each half covers roughly half the parent's variance.
        scale_reduction = 0.5 * math.log(2.0)  # ≈ 0.347
        self.scaling.data[parent_idcs] -= scale_reduction
        self.scaling.data[child_slice] = self.scaling.data[parent_idcs]

        self.num_active.fill_(old_n + num_to_add)
        print(f"  Coarse-to-fine: activated {num_to_add} Gaussians "
              f"({old_n} → {old_n + num_to_add})")

    def initialize_parameters_from_image(self, stack_imgs, stack_affines, bbox_world, lambda_init=0.3, device=None, num_to_init=None):
        """
        Content-adaptive initialisation for mu and color.

        Gaussian means are sampled from locations weighted by gradient magnitude
        across all input stacks, concentrating primitives at tissue boundaries.
        Follows the sampling strategy of Image-GS (Section 3.3, Eq. 6).

        Args:
            stack_imgs:    List of (H, W, D) normalised numpy arrays.
            stack_affines: (S, 4, 4) NIfTI affine matrices (voxel → world).
            bbox_world:    [min, max] world-space bounding box (unused currently).
            lambda_init:   Mixture weight for uniform sampling; 0 = pure
                           gradient-weighted, 1 = pure uniform.
            device:        Torch device for the initialised parameters.
            num_to_init:   Number of Gaussians to initialise (default: all).
                           When coarse-to-fine is enabled, only the first
                           num_to_init slots are content-adaptively initialised.
        """
        if num_to_init is None:
            num_to_init = self.num_gaussians
        print(f"Running content-adaptive parameter initialization ({num_to_init} Gaussians)...")
        grad_mags = []
        grad_mag_coord_wrlds = []
        values_nz = []
        for stack_img, stack_affine in zip(stack_imgs, stack_affines):
            # 1. Calculate Gradient Magnitude
            # Per-axis np.gradient with singleton-axis guard so this works on
            # both multi-slice stacks and single-slice (H, W, 1) inputs (e.g.
            # SVoRT-aligned slices). np.gradient errors on axes of size 1, so
            # we replace those with a zero gradient.
            values = stack_img.ravel()
            grads = [np.gradient(stack_img, axis=ax) if stack_img.shape[ax] > 1
                     else np.zeros_like(stack_img)
                     for ax in range(stack_img.ndim)]
            grad_mag = np.sqrt(sum(g**2 for g in grads)) # ||∇I(x)||_2


            grad_mag_voxel = np.where(grad_mag > 0)
            grad_mag_voxel = np.stack(grad_mag_voxel, axis=-1)
            grad_mag_coord_wrld = np.einsum('ij, nj -> ni', stack_affine[:3, :3], grad_mag_voxel) + stack_affine[:3, 3]
            grad_mag_coord_wrlds.append(grad_mag_coord_wrld)
            grad_mag_flat = grad_mag.ravel()
            mask = (grad_mag_flat > 0) & (values > 0)
            grad_mag_flat_nz = grad_mag_flat[mask]
            grad_mags.append(grad_mag_flat_nz)
            values_nz.append(values[mask])

        grad_mags = np.concatenate(grad_mags, axis=0)
        grad_mag_coord_wrlds = np.concatenate(grad_mag_coord_wrlds, axis=0)
        values_nz = np.concatenate(values_nz, axis=0)

        # 2. Calculate Sampling Probabilities (Eq. 6) 
        grad_sum = np.sum(grad_mags)
        if grad_sum > 0:
            grad_prob = grad_mags / grad_sum
        else: # Handle flat/empty image
            grad_prob = np.zeros_like(grad_mags)

        uniform_prob = 1.0 / grad_mags.size
        P_init = (1.0 - lambda_init) * grad_prob + lambda_init * uniform_prob
        P_init /= np.sum(P_init) # Ensure it sums to 1    
        # Sample indices based on the probability distribution.
        # replacement=True avoids a crash when total gradient voxels < num_to_init.
        sampled_indices = torch.multinomial(torch.from_numpy(P_init),
                                            num_samples=num_to_init,
                                            replacement=True)
        sampled_coords_wrld = torch.from_numpy(grad_mag_coord_wrlds[sampled_indices]).to(dtype=torch.float32, device=device)
        sampled_colors = torch.from_numpy(values_nz[sampled_indices]).to(dtype=torch.float32, device=device)

        # 5. Initialize Parameters
        # Store raw_color as softplus_inverse(sampled_colors) so that
        # softplus(raw_color) ≈ sampled_colors at initialisation.
        raw = torch.log(torch.expm1(sampled_colors.clamp(min=1e-6)))
        with torch.no_grad():
            self.mu.data[:num_to_init] = sampled_coords_wrld.to(device)
            self.raw_color.data[:num_to_init] = raw.to(device)

    def forward(self, coords, top_k_idcs, cov_psf=None):
        """
        Evaluates the Gaussian field at query coordinates using the K nearest
        primitives.  The PSF covariance is analytically folded into each
        Gaussian via covariance addition (exact for Gaussian ★ Gaussian).

        Args:
            coords:      (N, 3) query coordinates in world space.
            top_k_idcs:  (N, K) indices of the K nearest primitives per point.
            cov_psf:     (N, 3, 3) per-point PSF covariance matrices, or None.

        Returns:
            intensity: (N,) predicted intensity at each query point.
        """
        # v = x - mu_k  →  (N, K, 3)
        x_minus_mu = coords.unsqueeze(1) - self.mu[top_k_idcs]

        # Gaussian covariances: (K_total, 3, 3) → gather to (N, K, 3, 3)
        # Computes covariances for all pre-allocated Gaussians (including
        # inactive c2f slots).  The cost is small relative to the N*K
        # Mahalanobis distance computation, and keeping the compute_sigma()
        # call shape-static avoids torch.compile graph breaks.
        covs = self.compute_sigma()
        covs = covs[top_k_idcs]

        # Add PSF covariance: exploits Gaussian convolution closure (Sigma_total = Sigma_g + Sigma_psf)
        if cov_psf is not None:
            covs = covs + cov_psf[:, None, :, :]

        # Squared Mahalanobis distances  →  (N, K)
        inner_term = fused_mahalanobis_distance(covs, x_minus_mu)

        # Unnormalised Gaussian weights
        spatial_weights = torch.exp(-0.5 * inner_term)  # (N, K)

        # Soft normalisation: delta prevents division by zero in empty space
        # and acts as a background suppression constant
        delta = 1e-3
        gaussian_weights_nrmd = spatial_weights / (spatial_weights.sum(dim=1, keepdim=True) + delta)

        intensity = torch.sum(gaussian_weights_nrmd * self.color[top_k_idcs], dim=1)
        return intensity

    def compute_sigma(self, indices=None):
        """
        Builds the covariance matrix Sigma = R S R^T for Gaussian primitives.

        When *indices* is provided, covariances are computed only for the
        specified subset — this avoids O(K_total) work when only a fraction of
        the primitives are active (coarse-to-fine) or referenced (per-batch).

        `scaling` stores log-standard-deviations: exp(scaling[i]) is the std in mm
        along principal axis i, and exp(2*scaling[i]) is the variance.  This is the
        standard 3DGS convention and matches config semantics (scale_target=0.5 means
        target std ≈ exp(0.5) ≈ 1.65 mm).

        Args:
            indices: Optional index tensor selecting a subset of primitives.
                     If None, covariances are computed for all K primitives.

        Returns:
            Sigma: (M, 3, 3) covariance matrices, where M = len(indices) or K.
        """
        rot = self.rotation_g if indices is None else self.rotation_g[indices]
        scl = self.scaling if indices is None else self.scaling[indices]
        R = build_rotation_matrix_from_quaternion(rot)
        clamped_scaling = torch.clamp(scl, min=-3.0, max=3.0)
        S = torch.diag_embed(torch.exp(2.0 * clamped_scaling))
        Sigma = torch.bmm(R, torch.bmm(S, R.transpose(1, 2)))
        return Sigma
