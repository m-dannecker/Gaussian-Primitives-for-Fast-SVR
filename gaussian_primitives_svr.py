from mpmath import si
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import time
from sympy import Q
import torch
import torch.nn as nn
import faiss
import random
import faiss.contrib.torch_utils # important so that faiss can use torch tensors
import scipy.ndimage as ndi
import warnings
import ants
import math
# Suppress the specific internal warning from torch.compile
warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")

torch.set_float32_matmul_precision('high')

# OUTPUT_DIR = '/data_ssd/users/danneckm/GaussianSplatting/output'

@torch.compile(mode="reduce-overhead")
def fused_mahalanobis_distance(sigmas_nk, x_minus_mu_nk):
    """
    Computes v^T * Sigma^-1 * v for batch N, K.
    Fuses 3x3 inversion and matrix multiplication into one kernel.
    
    sigmas_nk: (N, K, 3, 3) - The covariance matrices (including PSF)
    x_minus_mu_nk: (N, K, 3) - The centered coordinates
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


@torch.compile(mode="reduce-overhead")
def fused_motion_correction_kernel(
    coords_n_3, 
    quats_k_4, 
    trans_k_3, 
    slice_ids_n, 
    slice_centers_n_3,
    sigma_psf_n_3_3=None
):
    """
    Fuses Quaternion->Matrix, Coordinate Transform, and Sigma Transform.
    """
    # 1. Gather Motion Parameters for each point N
    # We must index inside the kernel to allow fusion of the gather + math
    # quats_n: (N, 4)
    q_w = quats_k_4[slice_ids_n, 0]
    q_x = quats_k_4[slice_ids_n, 1]
    q_y = quats_k_4[slice_ids_n, 2]
    q_z = quats_k_4[slice_ids_n, 3]
    
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
    rot_x = r00 * c_x + r01 * c_y + r02 * c_z
    rot_y = r10 * c_x + r11 * c_y + r12 * c_z
    rot_z = r20 * c_x + r21 * c_y + r22 * c_z

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

    return coords_mc, sigma_tf

# Gaussian Primitives SVR
class GSVR(nn.Module):
    def __init__(self, num_gaussians, num_slices, D, bbox_wrld, mc=False, apply_slice_scaling=False, apply_slice_uncertainty=False):
        super().__init__()
        # Ensure D=3 for this 3D model
        self.D = 3
        self.bbox_wrld = bbox_wrld
        self.mc = mc
        if D != 3:
            print(f"Warning: Initializing 3D Gaussian Primitives SVR model but D was set to {D}. Forcing D=3.")
            
        self.num_gaussians = num_gaussians # K
        self.sigmoid = nn.Sigmoid() 
        # --- 3D Parameters ---
        
        # 1. Mean (mu)
        # Initialize mean randomly across the space, -1 to 1
        self.mu = nn.Parameter((torch.rand(num_gaussians, self.D) * 2 - 1) * (bbox_wrld.cpu()//2-1)) 
        # 2. Scaling (s)
        self.scaling = nn.Parameter((torch.ones(num_gaussians, self.D) + torch.randn(num_gaussians, self.D)*0.1) * 0.1)
        # 3. Rotation (q) - Quaternions
        # (K, 4) tensor. (w, x, y, z)
        # Initialize to identity rotation (w=1, x=0, y=0, z=0)
        self.rotation_g = nn.Parameter(torch.zeros(num_gaussians, 4))
        self.rotation_g.data[:, 0] = 1.0 
        # 4. Color (c)
        self.color = nn.Parameter(torch.ones(num_gaussians) * 0.5)

        # MOTION CORRECTION PARAMETERS
        self.rotation_mc = nn.Parameter(torch.zeros(num_slices, 4)) if mc else None
        if mc: self.rotation_mc.data[:, 0] = 1.0
        self.translation_mc = nn.Parameter(torch.zeros(num_slices, 3)) if mc else None
        if apply_slice_scaling:
            self.slice_scaling = nn.Parameter(torch.ones(num_slices))
            # SLICE UNCERTAINTY (Aleatoric Loss)
        if apply_slice_uncertainty:
            # Initialize to 0 (variance = 1)
            self.slice_weight = nn.Parameter(torch.ones(num_slices))

    def initialize_parameters_from_image(self, stack_imgs, stack_affines, lambda_init=0.3, device=None):
        """
        Content-adaptive initialization for mu and color, based on
        Section 3.3 and Equation (6) of the Image-GS paper. [cite: 230, 231, 234]
        
        Args:
            target_image_np (np.ndarray): The target image volume.
            lambda_init (float): Weight for uniform sampling. [cite: 238]
        """
        print("Running content-adaptive parameter initialization...")
        grad_mags = []
        grad_mag_coord_wrlds = []
        values_nz = []
        for stack_img, stack_affine in zip(stack_imgs, stack_affines):
            # 1. Calculate Gradient Magnitude
            # np.gradient returns a list of arrays [grad_d0, grad_d1, ...]
            values = stack_img.ravel()
            grads = np.gradient(stack_img)
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

            # save grad_mag to file as matplotlib figure for slice shape[0]//2, :, :
            plt.imshow(grad_mag[stack_img.shape[0]//2, :, :], cmap='gray')
            plt.savefig('grad_mag.png')
            plt.close()

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
        # Sample 'num_gaussians' indices based on the probability distribution
        sampled_indices = torch.multinomial(torch.from_numpy(P_init), 
                                            num_samples=self.num_gaussians, 
                                            replacement=False)
        sampled_coords_wrld = torch.from_numpy(grad_mag_coord_wrlds[sampled_indices]).to(dtype=torch.float32, device=device)
        sampled_colors = torch.from_numpy(values_nz[sampled_indices]).to(dtype=torch.float32, device=device)

        # 5. Initialize Parameters
        with torch.no_grad():
            self.mu.data = sampled_coords_wrld.to(device)
        
        # colors_nrmd_sigmoid_inv = -torch.log(1.0 / (sampled_colors+1e-6) - 1.0)
        # colors_softplus_inv = torch.log(torch.exp(sampled_colors) - 1.0)
        with torch.no_grad():
            # self.color.data = colors_nrmd_sigmoid_inv.to(device)
            self.color.data = sampled_colors.to(device)

    def forward(self, coords, slice_ids=None, slice_centers=None, Sigma_psf=None, top_k_idcs=None, scale_scale=1.0, scale_threshold=1.0):
        '''
            coords: (N, D) tensor of coordinates, where D=3
            slice_centers: (N, D) tensor of slice centers, where D=3
            slice_idcs: (N,) tensor of slice indices
        '''
        # K = num_gaussians
        # N = number of coordinates
        # D = 3
        
        q = None
        t = None
        # --- Compute Gaussian ---
        # 1. Compute v = (x - mu)
        # (N, 1, D) - (1, K, D) -> (N, K, D)
        mus = self.mu[top_k_idcs] if top_k_idcs is not None else self.mu
        # mus = self.mu
        x_minus_mu = (coords.unsqueeze(1) - mus)

        # 2. Compute Inverse Covariance
        sigmas = self.compute_sigma(scale_scale)     # (K, D, D)
        sigmas = sigmas[top_k_idcs]

        if Sigma_psf is not None: # use PSF, meaning Sigma_new = Sigma + Sigma_psf, Sigma_new_inv = torch.linalg.inv(Sigma_new)
            # TODO: I think Sigma_psf needs to be transformed by the motion correction parameters before adding it to the sigma?
            sigmas = (sigmas + Sigma_psf[:, None, :, :]).to(dtype=torch.float32)
        else:
            sigmas = sigmas.to(dtype=torch.float32)

        # 3. Compute the inner term: v^T @ M @ v
        # use torch.linalg.solve instead of torch.linalg.inv for better numerical stability and speed
        # M_v = Sigma_inv @ (x_minus_mu) <=> Sigma @ M_v = x_minus_mu --> torch.linalg.solve(Sigma, x_minus_mu) = M_v
        # sigmas_flat = sigmas.view(-1, 3, 3).to(dtype=torch.float32)
        # v_flat = x_minus_mu.view(-1, 3).to(dtype=torch.float32)

        # linalg solve
        # M_v_flat = torch.linalg.solve(sigmas_flat, v_flat)
        # M_v = M_v_flat.view(sigmas.shape[0], sigmas.shape[1], 3)
        # inner_term = torch.sum(x_minus_mu * M_v, dim=2) 
        

        # try cholesky decomposition instead of inverse
        # L = torch.linalg.cholesky(sigmas_flat)
        # y = torch.linalg.solve_triangular(L, v_flat.unsqueeze(-1), upper=False).squeeze(-1)
        # inner_term = torch.sum(y**2, dim=1).view(sigmas.shape[0], sigmas.shape[1])

        inner_term = fused_mahalanobis_distance(sigmas, x_minus_mu)
        
        # 4. Compute weights
        spatial_weights = torch.exp(-0.5 * inner_term) # (N, K)
        # 3. Learned Opacity
        # alpha_logits = torch.sigmoid(self.logit_opacity[top_k_idcs]) 
        
        # 4. Color / Luminance
        # We clamp to 0.0 to ensure no negative colors, but NO min_luminance floor needed now.
        color_vals = self.color[top_k_idcs]
        # color_vals = color_vals.clamp(min=0.3, max=1.0)
        
        # --- THE IMPLEMENTATION OF YOUR IDEA ---
        # "Luminance Gating": The effective opacity is scaled by the brightness.
        # If Color -> 0 (Background), then Weight -> 0 (Transparent).
        # This acts as a continuous, differentiable mask.
        raw_weights = spatial_weights# * alpha_logits * color_vals

        # 5. Soft Normalization (Still essential for noise reduction!)
        # We still need delta to handle the "empty space" math
        delta = 5e-2#6
        sum_weights = torch.sum(raw_weights, dim=1, keepdim=True)
        gaussian_weights_nrmd = raw_weights / (sum_weights + delta)
        # 6. Final Composition
        # Note: Since weights are already scaled by color, we just sum them?
        # NO. We need to be careful. Standard splatting is Sum(w_i * c_i).
        # Since we put color into w_i, we have effectively squared the color influence for the weighing
        # but we must still multiply by color for the final value.
        
        intensity = torch.sum(gaussian_weights_nrmd * color_vals, dim=1)
        if scale_scale != 1.0:
            intensity = intensity * (sum_weights.squeeze(-1) >scale_threshold).float()

        # 7. Background Composite
        # Calculate how much "signal" is here.
        # alpha_acc = sum_weights / (sum_weights + delta)
        # bg_color = 0.0
        
        # final_color = rgb_splat + (1.0 - alpha_acc.squeeze(-1)) * bg_color
        return intensity, q, t,

    def motion_correction_fused(self, coords, slice_ids, slice_centers, Sigma_psf=None):
        '''
        Applies motion correction to the coordinates.
        coords: (N, D) tensor of coordinates, where D=3
        slice_idcs: (N,) tensor of slice indices
        slice_centers: (N, D) tensor of slice centers, where D=3
        '''
        # coords_min = coords.min(dim=0).values
        # coords_max = coords.max(dim=0).values
        # coords_range = coords_max - coords_min
        # coords_nrmd = 2 * ((coords - coords_min) / coords_range) - 1
        # slice_centers_nrmd = 2 * (slice_centers / coords_range)
        quats = self.rotation_mc
        trans = self.translation_mc

        # Call the fused kernel
        coords_mc, Sigma_psf_tf = fused_motion_correction_kernel(
            coords, quats, trans, slice_ids, slice_centers, Sigma_psf
        )

        # For the regularization loss, we still need to fetch the specific q/t
        # But this is cheap compared to the transform
        batch_q = quats[slice_ids]
        batch_t = trans[slice_ids]
        rot = batch_q[:, 1:] # x,y,z parts for visualization if needed
        return coords_mc, Sigma_psf_tf, rot, batch_t

    def motion_correction(self, coords, slice_ids, slice_centers, Sigma_psf=None):
        '''
        Applies motion correction to the coordinates.
        coords: (N, D) tensor of coordinates, where D=3
        slice_idcs: (N,) tensor of slice indices
        slice_centers: (N, D) tensor of slice centers, where D=3
        '''
        # coords_min = coords.min(dim=0).values
        # coords_max = coords.max(dim=0).values
        # coords_range = coords_max - coords_min
        # coords_nrmd = 2 * ((coords - coords_min) / coords_range) - 1
        # slice_centers_nrmd = 2 * (slice_centers / coords_range)
        quats = self.rotation_mc[slice_ids]
        # 1. Compute the rotation matrix
        R = self.build_rotation_matrix_from_quaternion(quats) # (K, 3, 3)
        # 2. Compute the translation
        t = self.translation_mc[slice_ids] # (N, 3)
        # 3. Apply the motion correction
        # 3.1 shift the coordinates to the slice center
        coords = coords - slice_centers
        # 3.2 apply the rotation
        coords_mc = torch.einsum('nij,nj->ni', R, coords) + t
        # 3.3 shift the coordinates back to the original position
        coords_mc = coords_mc + slice_centers
        Sigma_psf_tf = torch.einsum('nij,njk->nik', R, Sigma_psf)
        Sigma_psf_tf = torch.einsum('nik,njk->nij', Sigma_psf_tf, R)
        rot = quats[:, 1:]
        return coords_mc, Sigma_psf_tf, rot, t

    def build_rotation_matrix_from_quaternion(self, q):
        '''
        Builds a batch of 3x3 rotation matrices from a batch of quaternions.
        q: (K, 4) tensor (w, x, y, z)
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

    def compute_sigma(self, scale_scale=1.0):
        '''
        Computes the inverse covariance matrix Sigma_inv = R @ S_inv @ R.T
        This is the compute-efficient way from the 3DGS paper.
        '''
        # 1. Get R from quaternions
        R = self.build_rotation_matrix_from_quaternion(self.rotation_g) # (K, 3, 3)

        # if inv:
        #     scaling_inv = 1 / self.scaling # (K, 3)
        #     S_inv = torch.diag_embed(scaling_inv**2) # (K, 3, 3)
        #     Sigma_inv = torch.bmm(R, torch.bmm(S_inv, R.transpose(1, 2))) # (K, 3, 3)
        #     return Sigma_inv
        # else: # use PSF, meaning Sigma_new = Sigma + Sigma_psf, Sigma_new_inv = torch.linalg.inv(Sigma_new)
        # scaling = self.scaling
        # S = torch.diag_embed(scaling**2) # (K, 3, 3)
        scaling = torch.exp(self.scaling)
        S = torch.diag_embed(scaling * scale_scale) # (K, 3, 3)
        Sigma = torch.bmm(R, torch.bmm(S, R.transpose(1, 2))) # (K, 3, 3)  
        return Sigma

def create_vis_grid(grid_shape, bbox_wrld, device):
    '''Creates a 3D coordinate grid.
    
    Args:
        grid_shape: (3,) tuple of the grid shape
        spacing: (3,) tuple of the voxel spacing
        device: torch.device
    Returns:
        coords_flat: (N, 3) tensor of coordinates
    '''
    coords_axes = [torch.linspace(-1, 1, s) * (b/2) for s, b in zip(grid_shape, bbox_wrld)]
    grid_x, grid_y, grid_z = torch.meshgrid(*coords_axes, indexing='ij')
    coords_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1)
    coords_flat = coords_flat.to(device)
    return coords_flat


def visualize_gaussians(groundtruth, gs, gpu_index, vis_grid_flat, shape_rec, spacing_rec, new_origin, epoch=0, k=10, BATCH_SIZE=1_000_000, psf_scale_fac=1.0, output_file_path=None, PSF=False, ttime=0, min_value=0.0, max_value=1.0, scale_scale=0.0, scale_threshold=1.0):
    '''
    Visualizes a 2D slice of the 3D Gaussian field using k-NN.
    gs: The Gaussian Primitives model
    vis_grid_flat: (N_vis, 3) tensor of coordinates for the slice
    vis_slice_shape: (H, W) tuple for reshaping the final image
    '''
    device = vis_grid_flat.device
    # 2. Evaluate the model
    values_pred_flat = torch.empty((vis_grid_flat.shape[0],), device=device)
    for step in range(0, vis_grid_flat.shape[0], BATCH_SIZE):
        with torch.no_grad():
            # --- 10. Use faiss for visualization culling ---
            _, top_k_idcs = gpu_index.search(vis_grid_flat[step:step+BATCH_SIZE], k)
            if (top_k_idcs >= gs.num_gaussians).any() or (top_k_idcs < 0).any():
                print("Error: Faiss returned invalid indices!")
                top_k_idcs = torch.clamp(top_k_idcs, min=0, max=gs.num_gaussians-1)
            Sigma_psf = generate_cov_psf(np.eye(3), spacing_rec, slice_thickness=None, scale_factor=psf_scale_fac) if PSF else None
            Sigma_psf = torch.tensor(np.broadcast_to(Sigma_psf, (len(top_k_idcs), 3, 3))).float().to(device) if Sigma_psf is not None else None
            values_pred_flat[step:step+BATCH_SIZE] = gs(vis_grid_flat[step:step+BATCH_SIZE], slice_ids=None, slice_centers=None, Sigma_psf=Sigma_psf, top_k_idcs=top_k_idcs, scale_scale=scale_scale, scale_threshold=scale_threshold)[0]
    # 3. Reshape and save
    values_pred = values_pred_flat.reshape(shape_rec).detach().cpu().numpy().clip(0.0, None)
    # unnormalize the values
    values_pred = values_pred * (max_value - min_value) + min_value

    groundtruth_data = groundtruth.get_fdata() if groundtruth is not None else None

    # values_pred = (values_pred - values_pred.min()) / (values_pred.max() - values_pred.min())
    # groundtruth_data = ((groundtruth_data - groundtruth_data.min()) / (groundtruth_data.max() - groundtruth_data.min())).astype(np.float32)


    # psnr = skimage.metrics.peak_signal_noise_ratio(groundtruth, values_pred)
    # ssim = skimage.metrics.structural_similarity(groundtruth, values_pred, data_range=1.0)
    psnr=0
    ssim=0
    rec_affine = np.diag(list(spacing_rec) + [1.0])
    rec_affine[:3, 3] = new_origin
    # save the reconstructed image as a 3D volume via nibabel
    rec_nii = nib.Nifti1Image(values_pred, rec_affine)
    filename = output_file_path.replace('.nii.gz', f'_ttime={ttime:.2f}s.nii.gz')
    nib.save(rec_nii, filename)

    if epoch == 0 and groundtruth is not None:
        # gt_aff = groundtruth.affine
        groundtruth_3D = nib.Nifti1Image(groundtruth_data, groundtruth.affine)
        nib.save(groundtruth_3D, output_file_path.replace('.nii.gz', f'_groundtruth_3D_epoch_{epoch}.nii.gz'))
    print(f"Epoch {epoch} PSNR: {psnr}, SSIM: {ssim}")

    # empty the cache
    torch.cuda.empty_cache()
    return values_pred


def _init_optim(gs, MC, SLICE_SCALING, SLICE_UNCERTAINTY, max_epochs, lrs):
    '''
    Initializes the optimizer.
    '''
    param_groups = [
        {'params': gs.mu, 'lr': lrs['position'], 'name': 'gs_mu'},
        {'params': gs.scaling, 'lr': lrs['scaling'], 'name': 'gs_scaling' },
        {'params': gs.rotation_g, 'lr': lrs['rotation'], 'name': 'gs_rotation_g'},
        {'params': gs.color, 'lr': lrs['color'], 'name': 'gs_color'},
    ]
    if MC:
        param_groups.append({'params': gs.rotation_mc, 'lr': lrs['motion_rot']})
        param_groups.append({'params': gs.translation_mc, 'lr': lrs['motion_trans']})
    if SLICE_SCALING:   
        param_groups.append({'params': gs.slice_scaling, 'lr': lrs['slice_scale'], 'name': 'gs_slice_scaling'})
    if SLICE_UNCERTAINTY:
        param_groups.append({'params': gs.slice_weight, 'lr': lrs['slice_weight'], 'name': 'gs_slice_weights'})

    # --- 3. Optimizer Setup ---
    optimizer = torch.optim.AdamW(param_groups)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1.0e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    grad_scaler = torch.amp.GradScaler()
    return optimizer, scheduler, grad_scaler


def train(stack_paths=None, mask_paths=None, config=None, output_file_path=None):
    # seed everything
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    lrs = config['training']['learning_rates']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_epochs = config['training']['max_epochs'] + 1
    num_gaussians = config['model']['num_gaussians']
    D = 3 # Dimension of our data (mu)
    M_NEIGHBORS = config['training']['neighbors']
    TOP_K_EVERY = config['training']['top_k_every']
    INIT = config['model']['init_type'] == 'content_adaptive'
    PSF = config['flags']['use_psf']
    MC = config['flags']['use_motion_correction']
    PSF_SCALE_FAC = config['reconstruction']['psf_scale_factor']
    SLICE_SCALING = config['flags']['use_slice_scaling']
    SLICE_UNCERTAINTY = config['flags']['use_slice_uncertainty']
    BIAS_FIELD_CORRECTION = config['preprocessing']['bias_field_correction']
    DENOISE = config['preprocessing']['denoise']
    spacing_rec = config['reconstruction']['spacing']
    slice_thickness = config['reconstruction']['slice_thickness']

    # --- ADD: Smoothness Regularization ---
    # We penalize scales for being far from our target "smooth" value
    l2_lambda = config['training']['loss_weights']['lambda_l2_scale']
    log_scale_target = config['training']['loss_weights']['log_scale_target']

    # --- 1. Data Setup ---
    (gt_nii, imgs, affines, slice_ids, slice_centers_wrld_np, 
    coords_wrld_np, values_np, Sigma_psf, bbox_wrld, new_origin, min_value, max_value) = load_data(stack_paths=stack_paths, mask_paths=mask_paths, slice_thickness=slice_thickness,
                                                                             psf_scale_fac=PSF_SCALE_FAC, dilate_mask_sigma=config['preprocessing']['dilate_mask_sigma'], 
                                                                             bias_field_correction=BIAS_FIELD_CORRECTION, denoise=DENOISE)
    shape_rec = [math.ceil(b/s) for b, s in zip(bbox_wrld, spacing_rec)]
    bbox_wrld = np.array([d*s for d, s in zip(shape_rec, spacing_rec)])
    
    # --- Pre-compute Visualization Grid ---
    vis_grid_flat_3D = create_vis_grid(shape_rec, bbox_wrld, device)
    print(f"Number of non-zero voxels in the raw data: {len(values_np)}")
    # --- 2. GSVR Model Setup ---
    gs = GSVR(num_gaussians, np.max(slice_ids)+1, D=D, bbox_wrld=torch.from_numpy(bbox_wrld).to(dtype=torch.float32, device=device), mc=MC, apply_slice_scaling=SLICE_SCALING, apply_slice_uncertainty=SLICE_UNCERTAINTY)
    if INIT:
        gs.initialize_parameters_from_image(imgs, affines, lambda_init=0.0, device=device)
    gs.to(device)
    print(f"Training on device: {device}")
    gs = torch.compile(gs, mode="reduce-overhead") # Can add this back for speed

    optimizer, scheduler, grad_scaler = _init_optim(gs, MC, SLICE_SCALING, SLICE_UNCERTAINTY, max_epochs, lrs)
    loss_fn = nn.L1Loss()
    # scheduler = None
    
    # --- 4. FAISS Setup ---
    ressource = faiss.StandardGpuResources() # Use GPU resources
    faiss_config = faiss.GpuIndexFlatConfig()
    faiss_config.device = 0
    # Create the GPU index directly (No CPU wrapper)
    gpu_index = faiss.GpuIndexFlatL2(ressource, D, faiss_config)
    # --- 5. Convert data to tensors and move to device ---
    coords_wrld = torch.from_numpy(coords_wrld_np).float().to(device)
    slice_centers_wrld = torch.from_numpy(slice_centers_wrld_np).float().to(device) if MC else None
    values = torch.from_numpy(values_np).float().to(device)
    Sigma_psf = torch.from_numpy(Sigma_psf).float().to(device) if PSF else None
    
    slice_ids = torch.from_numpy(slice_ids).int().to(device).contiguous()

    # --- 6. Training Loop ---
    t0 = time.time()
    total_time = 0
    for i in range(max_epochs):
        # batches
        batch_coords_wrld = coords_wrld.contiguous()#[0:len_data+BATCH_SIZE]
        batch_slice_centers_wrld = slice_centers_wrld#[0:len_data+BATCH_SIZE] if MC else None
        batch_values = values#[0:len_data+BATCH_SIZE]
        batch_Sigma_psf = Sigma_psf#[0:len_data+BATCH_SIZE] if PSF else None
        batch_slice_ids = slice_ids#[0:len_data+BATCH_SIZE] if MC else None

        loss_history = {'sr': [], 'sr_reg': [], 'mc': []}
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            if MC and batch_slice_centers_wrld is not None and batch_slice_ids is not None:
                batch_coords_wrld, batch_Sigma_psf_tf, batch_q, batch_t = gs.motion_correction_fused(batch_coords_wrld, batch_slice_ids, batch_slice_centers_wrld, Sigma_psf=batch_Sigma_psf)
                batch_Sigma_psf = batch_Sigma_psf_tf if batch_Sigma_psf_tf is not None else batch_Sigma_psf
            else:
                batch_q = torch.tensor([0.0])
                batch_t = torch.tensor([0.0])
            if i % TOP_K_EVERY == 0:
                gpu_index.reset()
                gpu_index.add(gs.mu.data)
                _, top_k_idcs = gpu_index.search(batch_coords_wrld, M_NEIGHBORS) # Search for nearest neighbors in the faiss index
                batch_top_k_idcs = top_k_idcs #torch.from_numpy(top_k_idcs).to(device)
                if (top_k_idcs >= gs.num_gaussians).any() or (top_k_idcs < 0).any():
                    print("Error: Faiss returned invalid indices!")
                    top_k_idcs = torch.clamp(top_k_idcs, min=0, max=gs.num_gaussians-1)

            color_pred, _, _ = gs(batch_coords_wrld, slice_ids=batch_slice_ids, slice_centers=batch_slice_centers_wrld, Sigma_psf=batch_Sigma_psf, top_k_idcs=batch_top_k_idcs)
            if SLICE_SCALING:
                slice_scales = gs.slice_scaling
                slice_scales_softplus = torch.nn.functional.softplus(slice_scales)
                slice_scales_softplus = slice_scales_softplus / slice_scales_softplus.mean()
                color_pred = color_pred * slice_scales_softplus[batch_slice_ids]
            if SLICE_UNCERTAINTY:
                slice_weights = gs.slice_weight[batch_slice_ids]
                sw_reg = 1e-1 * -(torch.log(slice_weights + 1e-8)).mean()
                l1_diff = torch.abs(color_pred - batch_values)
                loss_sr = (l1_diff * slice_weights).mean() + sw_reg
                l1_diff = l1_diff.mean()
            else:           
                loss_sr = loss_fn(color_pred, batch_values)
                l1_diff = loss_sr
            loss_mc = (batch_q**2).mean() + (batch_t**2).mean()
            loss_sr_reg = l2_lambda * ((gs.scaling - log_scale_target)**2).mean()
            loss = loss_sr + loss_sr_reg
            
        loss_history['sr'].append(l1_diff.item())
        loss_history['sr_reg'].append(loss_sr_reg.item())
        loss_history['mc'].append(loss_mc.item())
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        gs.color.data.clamp_(min=0.0, max=None)
        
        loss_mean = {'sr': np.array(loss_history['sr']).mean(), 'sr_reg': np.array(loss_history['sr_reg']).mean(), 'mc': np.array(loss_history['mc']).mean()}    
        if scheduler is not None:
            scheduler.step() # Step scheduler once per epoch
        if i == 0:
            t0 = time.time() # don't count the very first epoch in the total training time due to torch compilation time 
        if i % 50 == 0 or (i == max_epochs-1):
            t1 = time.time()
            t_delta = t1-t0
            total_time += t_delta
            print(f'Epoch {i} loss sr: {loss_mean["sr"]:.6f}, loss sr_reg: {loss_mean["sr_reg"]:.6f}, loss mc: {loss_mean["mc"]:.6f}, time per epoch: {t_delta/50:.2f}s, total training time: {total_time:.2f}s')
            t0 = time.time()
        if (i==max_epochs-1): # or (i % 500 == 0) and (i > 0):
            print(f"Visualizing epoch {i}...")
            visualize_gaussians(gt_nii, gs, gpu_index, vis_grid_flat_3D, shape_rec, spacing_rec, new_origin, epoch=i, k=M_NEIGHBORS, psf_scale_fac=1.0, min_value=min_value, max_value=max_value, output_file_path=output_file_path, PSF=PSF, ttime=total_time, scale_scale=1.0, scale_threshold=0.01)
            t0 = time.time() # don't count visualization time in the total training time

def generate_cov_psf(affine3x3, spacing, slice_thickness=None, scale_factor=1.0):
    '''
    Generates a PSF covariance matrix for a given stack. 
    The psf is approximated as a 3D Gaussian with a given standard deviation.
    In-plane resolution is given by stack_spacing, slice thickness is given by slice_thickness if provided, otherwise it is assumed to be isotropic.
    '''
    fwhm_to_std = 1.2 / (2 * np.sqrt(2 * np.log(2))) * scale_factor
    sigma_psf_diag = np.array(spacing) * fwhm_to_std
    if slice_thickness is not None:
        sigma_psf_diag[2] = slice_thickness * fwhm_to_std / 1.2
    Sigma_psf = np.diag(sigma_psf_diag**2)
    # project it into the world space doing L@S@L.T where L is the normalized affine3x3
    L = affine3x3[:3, :3] / np.linalg.norm(affine3x3[:3, :3], axis=0)
    Sigma_psf = L @ Sigma_psf @ L.T
    return Sigma_psf
    

def dilate_mask(mask_np, sigma=1.0):
    '''
    Dilates a binary mask by a given sigma. If sigma is 0.0, returns the original mask.
    '''
    if sigma > 0.0:
        mask = mask_np.astype(np.float32)
        mask = (ndi.gaussian_filter(mask, sigma=sigma) > 0.001).astype(np.uint8)
        return mask
    else:
        return mask_np.astype(np.uint8)


def load_data(stack_paths=None, mask_paths=None, slice_thickness=None, psf_scale_fac=1.0, dilate_mask_sigma=0.5, bias_field_correction=False, denoise=False):
    stacks_img = stack_paths
    stacks_mask = mask_paths
    slice_thickness = [slice_thickness[0]]*len(stack_paths) if slice_thickness else slice_thickness
    gt_nii = None

    all_values = []
    all_coords_wrld = []
    all_affines = []
    all_imgs = []
    all_Sigma_psf = []
    all_slice_centers_wrld = [] # slice center where slice refers to the thick-slice, i.e. lat axis
    all_slice_idcs = []
    n_slices_global = 0
    percentile_99 = 0.0
    for i, stack in enumerate(stacks_img):
        # load images
        stack_img = nib.load(stack)
        stack_mask_data = ((stack_img.get_fdata()>0.01).astype(np.int32)) if len(stacks_mask) == 0 else (nib.load(stacks_mask[i]).get_fdata()>0).astype(np.int32).squeeze()
        stack_img_data = (stack_img.get_fdata() * stack_mask_data).clip(0.0, None)
        stack_mask_data = dilate_mask(stack_mask_data, sigma=dilate_mask_sigma)
        stack_affine = stack_img.affine
        stack_spacing = stack_img.header.get_zooms()[:3]
        # get 99th percentile of stack_img_data
        percentile_99 = max(percentile_99, np.percentile(stack_img_data[stack_mask_data>0], 99.9))
        # bias field correction
        # apply ants n4 bias field correction
        if bias_field_correction:
            print(f"Applying bias field correction to stack {i}...")
            stack_img_data = ants.n4_bias_field_correction(ants.from_nibabel(nib.Nifti1Image(stack_img_data, stack_affine)))
            if not denoise:
               stack_img_data = stack_img_data.to_nibabel().get_fdata()
        if denoise:
            if not bias_field_correction:
                stack_img_data = ants.from_nibabel(nib.Nifti1Image(stack_img_data, stack_affine))
            print(f"Denoising stack {i}...")
            stack_img_data = ants.denoise_image(stack_img_data).to_nibabel().get_fdata()
        # get values and coordinates in world space
        coords_vxl = np.argwhere(stack_mask_data>0)
        coords_vxl_homo = np.hstack([coords_vxl, np.ones((len(coords_vxl), 1))])
        coords_wrld = (stack_affine @ coords_vxl_homo.T).T[:, :3]

        # get slice centers in world space
        nx, ny, nz = stack_img_data.shape[:3]
        slice_axis_centers = np.array([nx / 2.0, ny / 2.0])
        slice_centers_voxel = np.array([np.hstack([slice_axis_centers, k, 1]) for k in range(nz)])
        slice_centers_world = (stack_affine @ slice_centers_voxel.T).T[:, :3]
        slice_indices = coords_vxl[:, 2].astype(int)
        slice_centers_wrld = slice_centers_world[slice_indices]
        slice_idcs_global = slice_indices + n_slices_global
        n_slices_global += nz

        # get PSF
        Sigma_psf = generate_cov_psf(stack_affine, stack_spacing, slice_thickness[i], scale_factor=psf_scale_fac)
        Sigma_psf = np.broadcast_to(Sigma_psf, (len(coords_wrld), 3, 3))
    
        # add to global lists
        all_imgs.append(stack_img_data)
        all_coords_wrld.append(coords_wrld)
        all_values.append(stack_img_data[coords_vxl[:, 0], coords_vxl[:, 1], coords_vxl[:, 2]])
        all_Sigma_psf.append(Sigma_psf)
        all_slice_centers_wrld.append(slice_centers_wrld)
        all_slice_idcs.append(slice_idcs_global)
        all_affines.append(stack_affine)

    all_values = np.concatenate(all_values, axis=0)
    all_coords_wrld = np.concatenate(all_coords_wrld, axis=0)
    all_slice_centers_wrld = np.concatenate(all_slice_centers_wrld, axis=0)
    all_slice_idcs = np.concatenate(all_slice_idcs, axis=0)
    all_Sigma_psf = np.concatenate(all_Sigma_psf, axis=0)
    global_center_of_mass = np.mean(all_coords_wrld, axis=0)
    all_coords_wrld = all_coords_wrld - global_center_of_mass
    all_slice_centers_wrld = all_slice_centers_wrld - global_center_of_mass
    all_affines = np.stack(all_affines, axis=0)
    all_affines[:, :3, 3] -= global_center_of_mass
    bbox_wrld = (np.max(all_coords_wrld, axis=0) - np.min(all_coords_wrld, axis=0)) * 1.1
    new_origin = np.min(all_coords_wrld, axis=0) + global_center_of_mass

    
    
    print(f"Global 99.9th percentile: {percentile_99}, global max: {all_values.max()}")
    all_values = all_values.clip(0.0, percentile_99)
    all_imgs = [img.clip(0.0, percentile_99) for img in all_imgs]
    max_value = all_values.max()
    min_value = 0
    all_values = (all_values - min_value) / (max_value - min_value)
    all_imgs = [(img - min_value) / (max_value - min_value) for img in all_imgs]

    return gt_nii, all_imgs, all_affines, all_slice_idcs, all_slice_centers_wrld, all_coords_wrld, all_values, all_Sigma_psf, bbox_wrld, new_origin, min_value, max_value


if __name__ == '__main__':
    # seed everything
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    train()