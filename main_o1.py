import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Mixed precision
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# --------------------- Custom Imports (Your Modules) ---------------------
from Model import Conv3DModel  # Your 3D CNN
from dataset import MedicalImageDataset  # Your training dataset
from dataset_patches import MedicalImageDataset_Patches  # Your training dataset

##############################################################################
#                          1. Cholesky Conversions (Torch-only)
##############################################################################
def lower_triangular_to_cholesky(tensor_elements: torch.Tensor) -> torch.Tensor:
    """
    Convert lower-triangular diffusion elements [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
    -> Cholesky factors [R0, R1, R2, R3, R4, R5], all in PyTorch (no NumPy).

    tensor_elements: shape [..., 6], e.g. [B, 6, H, W, Z] if channel=6.
    returns: shape [..., 6].
    """
    epsilon = 1e-10

    # Unpack
    Dxx = tensor_elements[..., 0]
    Dxy = tensor_elements[..., 1]
    Dyy = tensor_elements[..., 2]
    Dxz = tensor_elements[..., 3]
    Dyz = tensor_elements[..., 4]
    Dzz = tensor_elements[..., 5]

    # Compute Cholesky
    R0 = torch.sqrt(torch.clamp(Dxx, min=epsilon))
    R3 = torch.where(R0 > epsilon, Dxy / R0, torch.zeros_like(Dxy))
    R1 = torch.sqrt(torch.clamp(Dyy - R3 ** 2, min=epsilon))
    R5 = torch.where(R0 > epsilon, Dxz / R0, torch.zeros_like(Dxz))
    R4 = torch.where(R1 > epsilon, (Dyz - R3 * R5) / R1, torch.zeros_like(Dyz))
    R2 = torch.sqrt(torch.clamp(Dzz - R4 ** 2 - R5 ** 2, min=epsilon))

    cholesky_elements = torch.stack([R0, R1, R2, R3, R4, R5], dim=-1)
    return cholesky_elements


def cholesky_to_lower_triangular(R: torch.Tensor) -> torch.Tensor:
    """
    Convert Cholesky factors [R0, R1, R2, R3, R4, R5]
    -> [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz], all in PyTorch.

    R: shape [..., 6], e.g. [B, 6, H, W, Z] if channel=6.
    returns: shape [..., 6].
    """
    R0 = R[..., 0]
    R1 = R[..., 1]
    R2 = R[..., 2]
    R3 = R[..., 3]
    R4 = R[..., 4]
    R5 = R[..., 5]

    Dxx = R0 ** 2
    Dxy = R0 * R3
    Dyy = R1 ** 2 + R3 ** 2
    Dxz = R0 * R5
    Dyz = R1 * R4 + R3 * R5
    Dzz = R2 ** 2 + R4 ** 2 + R5 ** 2

    return torch.stack([Dxx, Dxy, Dyy, Dxz, Dyz, Dzz], dim=-1)


##############################################################################
#                          2. Vectorized Signal Calculation
##############################################################################
import torch


def fwdti_signal_prediction_lower_tri(
        S0: torch.Tensor,
        f: torch.Tensor,
        b,
        Dg: torch.Tensor,
        D_iso,
        g: torch.Tensor
) -> torch.Tensor:
    """
    Free-water DTI signal prediction with the diffusion tensor in
    lower triangular order:
       (Dxx, Dyx, Dyy, Dzx, Dzy, Dzz).

    We allow:
      - 'b' to be a float, 1D, 2D, or 5D tensor,
      - 'D_iso' to be float, 4D or 5D tensor, etc.

    The final shape of the output is [B, H, W, Z, G].
    """

    # ------------------------------------------------------------------
    # 0) Convert 'b' and 'D_iso' to torch.Tensor (if they aren't already).
    #
    #    This ensures we can do b.dim() or D_iso.dim() safely.
    # ------------------------------------------------------------------
    # Match the device/dtype of S0 if possible:
    device = S0.device
    dtype = S0.dtype

    b = torch.as_tensor(b, dtype=dtype, device=device)
    D_iso = torch.as_tensor(D_iso, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # 1) Unpack shape from S0 => [B,H,W,Z] (possibly with an extra dim)
    # ------------------------------------------------------------------
    B, H, W, Z = S0.shape[:4]  # e.g. if S0 is [B,H,W,Z], that's fine
    G_ = g.shape[-2]  # from g => [B,H,W,Z,G,3]

    # ------------------------------------------------------------------
    # 2) Broadcast/reshape 'b' to [B,H,W,Z,G]
    #
    # Common cases:
    #   - b is scalar => shape [] => broadcast -> [B,H,W,Z,G]
    #   - b is [G] => broadcast -> [B,H,W,Z,G]
    #   - b is [B,G] => broadcast -> [B,H,W,Z,G]
    #   - b is already [B,H,W,Z,G]
    # ------------------------------------------------------------------
    if b.dim() == 0:
        # single scalar => [] => broadcast
        # => shape [1,1,1,1,1] => expand to [B,H,W,Z,G]
        b = b.view(1, 1, 1, 1, 1).expand(B, H, W, Z, G_)
    elif b.dim() == 1 and b.shape[0] == G_:
        # shape [G], expand to [B,H,W,Z,G]
        b = b.view(1, 1, 1, 1, G_).expand(B, H, W, Z, G_)
    elif b.dim() == 2 and b.shape == (B, G_):
        # shape [B,G], expand to [B,H,W,Z,G]
        b = b.view(B, 1, 1, 1, G_).expand(B, H, W, Z, G_)
    elif b.dim() == 5 and b.shape == (B, H, W, Z, G_):
        # already correct shape
        pass
    else:
        raise ValueError(
            f"'b' has shape {b.shape}, which cannot be broadcast to [B,H,W,Z,G] = "
            f"[{B},{H},{W},{Z},{G_}]"
        )

    # ------------------------------------------------------------------
    # 3) Broadcast/reshape 'D_iso' to [B,H,W,Z,G]
    #
    # Common cases:
    #   - D_iso is scalar => shape [] => broadcast
    #   - D_iso is [B,H,W,Z] => broadcast -> [B,H,W,Z,G]
    #   - D_iso is [B,H,W,Z,G]
    # ------------------------------------------------------------------
    if D_iso.dim() == 0:
        # single scalar => [] => shape [1,1,1,1,1] => expand to [B,H,W,Z,G]
        D_iso = D_iso.view(1, 1, 1, 1, 1).expand(B, H, W, Z, G_)
    elif D_iso.dim() == 4 and D_iso.shape == (B, H, W, Z):
        # shape [B,H,W,Z] => unsqueeze => [B,H,W,Z,1] => expand to [B,H,W,Z,G]
        D_iso = D_iso.unsqueeze(-1).expand(B, H, W, Z, G_)
    elif D_iso.dim() == 5 and D_iso.shape == (B, H, W, Z, G_):
        # already correct shape
        pass
    else:
        raise ValueError(
            f"'D_iso' has shape {D_iso.shape}, which cannot be broadcast to [B,H,W,Z,G] = "
            f"[{B},{H},{W},{Z},{G_}]"
        )

    # ------------------------------------------------------------------
    # 4) Expand S0, f to have a trailing dimension of G => [B,H,W,Z,G]
    # ------------------------------------------------------------------
    while S0.dim() < 5:
        S0 = S0.unsqueeze(-1)  # => [B,H,W,Z,1]
    while f.dim() < 5:
        f = f.unsqueeze(-1)  # => [B,H,W,Z,1]

    if S0.shape[-1] == 1:
        S0 = S0.expand(-1, -1, -1, -1, G_)
    if f.shape[-1] == 1:
        f = f.expand(-1, -1, -1, -1, G_)

    # ------------------------------------------------------------------
    # 5) Extract Dxx, Dxy, Dyy, Dxz, Dyz, Dzz in lower-tri order
    #    Dg => [B,H,W,Z,6]
    #    We'll unsqueeze => [B,H,W,Z,1], which can broadcast over G
    # ------------------------------------------------------------------
    Dxx = Dg[..., 0].unsqueeze(-1)
    Dxy = Dg[..., 1].unsqueeze(-1)  # Dyx => rename to Dxy
    Dyy = Dg[..., 2].unsqueeze(-1)
    Dxz = Dg[..., 3].unsqueeze(-1)  # Dzx => rename to Dxz
    Dyz = Dg[..., 4].unsqueeze(-1)  # Dzy => rename to Dyz
    Dzz = Dg[..., 5].unsqueeze(-1)

    # ------------------------------------------------------------------
    # 6) Extract gradient components
    #    g => [B,H,W,Z,G,3]
    # ------------------------------------------------------------------
    g_x = g[..., 0]
    g_y = g[..., 1]
    g_z = g[..., 2]

    # ------------------------------------------------------------------
    # 7) Compute g^T D g => shape [B,H,W,Z,G]
    # ------------------------------------------------------------------
    gDg = (
            Dxx * g_x * g_x +
            2.0 * Dxy * g_x * g_y +
            Dyy * g_y * g_y +
            2.0 * Dxz * g_x * g_z +
            2.0 * Dyz * g_y * g_z +
            Dzz * g_z * g_z
    )

    # ------------------------------------------------------------------
    # 8) Exponent terms
    # ------------------------------------------------------------------
    b_gDg = b * gDg  # => [B,H,W,Z,G]
    exp_tissue = torch.exp(-b_gDg)
    exp_free = torch.exp(-b * D_iso)

    # ------------------------------------------------------------------
    # 9) Final signal
    # ------------------------------------------------------------------
    signal = S0 * ((1.0 - f) * exp_tissue + f * exp_free)
    return signal



##############################################################################
#                          3. Save as NIfTI
##############################################################################
# def save_tensor_to_nifti(tensor, base_filename):
#     """
#     Save a batch of 3D tensors as NIfTI files.
#     If multiple channels, we save the mean across them.
#     """
#     if tensor.is_cuda:
#         tensor = tensor.cpu()
#     for i in range(tensor.shape[0]):
#         # Suppose shape is [B, C, H, W, Z]
#         if tensor.shape[1] > 1:
#             img_data = tensor[i].mean(dim=0).numpy()
#         else:
#             img_data = tensor[i, 0].numpy()
#         nifti_img = nib.Nifti1Image(img_data, affine=np.eye(4))
#         filename = f"{base_filename}_{i}.nii.gz"
#         nib.save(nifti_img, filename)


def save_tensor_to_nifti(tensor, base_filename):
    """
    Save a batch of 3D or 4D tensors as NIfTI files.

    Expects tensor shape: [B, C, H, W, Z]
      - B: batch size
      - C: channel dimension (e.g. 6 for DTI, 1 for f)
      - H, W, Z: spatial dimensions

    The saved NIfTI will have shape [H, W, Z, C] per batch item.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Iterate over the batch dimension
    for i in range(tensor.shape[0]):
        # For the i-th element in the batch, shape is [C, H, W, Z]
        data_4d = tensor[i]  # [C, H, W, Z]
        # data_4d = data_4d.permute(1, 2, 3, 0)  # => [H, W, Z, C]

        # Convert to a NumPy array
        img_data = data_4d.numpy()

        # Create a NIfTI image with an identity affine
        nifti_img = nib.Nifti1Image(img_data, affine=np.eye(4))

        # Construct a filename, e.g. "base_0.nii.gz", "base_1.nii.gz", etc.
        filename = f"{base_filename}_{i}.nii.gz"
        nib.save(nifti_img, filename)
        # print(f"Saved: {filename}  (shape={img_data.shape})")


def oppsite_penalty_loss(y_pred, y_true, penalty_weight=100):
    opposite_sign_mask = (y_pred * y_true < 0).float()

    # Calculate the penalty for all elements where signs are opposite
    penalties = penalty_weight * opposite_sign_mask

    # Compute the mean of the penalties
    mean_penalty = torch.mean(penalties)

    return mean_penalty
##############################################################################
#                          4. Training & Validation Routines
##############################################################################
def train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        criterion_pinn,
        scaler,
        scheduler,
        epoch,
        epochs,
        device,
):
    """
    Single epoch of training with mixed precision.

    We assume each batch returns:
      cropped_sh:           [B, C, H, W, Z]
      cropped_mask:         [B, 1, H, W, Z]
      cropped_cholesky:     [B, 6, H, W, Z]
      cropped_f:            [B, 1, H, W, Z]
      cropped_dti:          [B, 6, H, W, Z]   (optional, not used in training here)
      cropped_si:           [B, G, H, W, Z]   (signal for G directions)
      cropped_s0:           [B, 1, H, W, Z]
      bval:                 [B, G]
      bvec:                 [B, 3, G]
    """
    model.train()
    running_loss = 0.0
    cf_loss_total = 0.0
    f_loss_total = 0.0
    pinn_loss_total = 0.0
    cos_loss_total = 0.0
    pena_loss_toal = 0.0
    start_time = time.time()

    for (
            cropped_sh,
            cropped_mask,
            cropped_cholesky_factors,
            cropped_f,
            cropped_dti,
            cropped_si,
            cropped_s0,
            bval,
            bvec
    ) in train_loader:
        # Move all to GPU
        cropped_sh = cropped_sh.permute(0, 4, 1, 2, 3).float().to(device)  # [B, C, H, W, Z]
        cropped_mask = cropped_mask.permute(0, 4, 1, 2, 3).float().to(device)  # [B, 1, H, W, Z]
        cropped_cholesky_factors = cropped_cholesky_factors.permute(0, 4, 1, 2, 3).float().to(device)
        cropped_f = cropped_f.permute(0, 4, 1, 2, 3).float().to(device)  # [B, 1, H, W, Z]
        cropped_si = cropped_si.permute(0, 4, 1, 2, 3).float().to(device)  # [B, G, H, W, Z]
        cropped_s0 = cropped_s0.permute(0, 4, 1, 2, 3).float().to(device)  # [B, 1, H, W, Z]
        cropped_dti = cropped_dti.permute(0, 4, 1, 2, 3).float().to(device)  # [B, 1, H, W, Z]
        # bval = bval.float().to(device)  # [B, G]
        bvec = bvec.float().to(device)  # [B, 3, G]

        optimizer.zero_grad()

        with autocast():
            # Forward through model: assume model expects [B, C, H, W, Z]
            predicted_cf_f = model(cropped_sh) * cropped_mask  # [B, 7, H, W, Z] => 6 cholesky + 1 f
            predicted_cf = predicted_cf_f[:, :6]  # => [B, 6, H, W, Z]
            predicted_f = predicted_cf_f[:, 6:]  # => [B, 1, H, W, Z]

            # Mask ground truth
            cropped_cholesky_factors *= cropped_mask
            cropped_f *= cropped_mask
            cropped_si *= cropped_mask  # If your data is 0 outside mask
            cropped_s0 *= cropped_mask

            # cropped_s0_mask = np.expand_dims((cropped_s0 > 0), axis=1)
            cropped_s0 = (cropped_s0 > 0) * cropped_s0
            cropped_si = (cropped_si > 0) * cropped_si

            # Basic L1 losses
            # cropped_cholesky_factors = torch.clamp(cropped_cholesky_factors, max=0.05477225575051661, min=0)

            cropped_cholesky_factors[:, 0] = torch.clamp(cropped_cholesky_factors[:, 0], max=0.05477225575051661, min=0)
            cropped_cholesky_factors[:, 1] = torch.clamp(cropped_cholesky_factors[:, 1], max=0.05477225575051661, min=0)
            cropped_cholesky_factors[:, 2] = torch.clamp(cropped_cholesky_factors[:, 2], max=0.05477225575051661, min=0)
            cropped_cholesky_factors[:, 3] = torch.clamp(cropped_cholesky_factors[:, 3], max=0.05, min=-0.05)
            cropped_cholesky_factors[:, 5] = torch.clamp(cropped_cholesky_factors[:, 5], max=0.05, min=-0.05)

            cropped_cholesky_factors[:, 4] = torch.clamp(cropped_cholesky_factors[:, 4], max=0.05, min=-0.05)

            loss_cf = criterion(predicted_cf, cropped_cholesky_factors)
            loss_pena = oppsite_penalty_loss(predicted_cf, cropped_cholesky_factors)

            # cosine_sim = F.cosine_similarity(predicted_cf, cropped_cholesky_factors, dim=1)
            # cosine_loss = (1 - cosine_sim).sum() * 0.001

            #
            # loss_cf_1 = criterion(predicted_cf[:,1:2], cropped_dti[:,1:2]) * 1
            # loss_cf_3 = criterion(predicted_cf[:, 3:4], cropped_dti[:, 3:4])
            # loss_cf_4 = criterion(predicted_cf[:, 4:5], cropped_dti[:, 4:5]) * 1
            #
            # loss_cf_0 = criterion(predicted_cf[:,0:1], cropped_dti[:,0:1])
            # loss_cf_2 = criterion(predicted_cf[:, 2:3], cropped_dti[:, 2:3])
            # loss_cf_5 = criterion(predicted_cf[:, 5:6], cropped_dti[:, 5:6])
            # loss_cf = loss_cf_0 + loss_cf_1 + loss_cf_2 + loss_cf_3 + loss_cf_4 + loss_cf_5


            loss_f = criterion(predicted_f, cropped_f) * 0.001


            if epoch > 40:
                # ~~~~~~~~~~~~~ PINN-like reconstruction ~~~~~~~~~~~~~
                # Dg = predicted_cf.permute(0, 2, 3, 4, 1)  # => [B, H, W, Z, 6]

                # Convert from cholesky to 6 unique D components
                pred_cf_perm = predicted_cf.permute(0, 2, 3, 4, 1)  # => [B, H, W, Z, 6]
                Dg = cholesky_to_lower_triangular(pred_cf_perm)  # => [B, H, W, Z, 6]


                S0 = cropped_s0.squeeze(1)
                f = predicted_f.squeeze(1)
                b_scalar = 400
                D_iso_float = 3.0e-3

                B, H, W, Z = S0.shape
                _, _, G_ = bvec.shape

                # g = bvec.permute(0, 2, 1).view(B,1,1,1,G_,3).expand(1, H, W, Z, 1, 1) # [B, 3, G] -> B, H, W, Z, G, 3
                g = bvec.permute(0, 2, 1)  # => [B, G_, 3]

                # Now shape => [B, 1, 1, 1, G_, 3]
                g = g.view(B, 1, 1, 1, G_, 3)

                # Finally broadcast over H, W, Z => [B, H, W, Z, G_, 3]
                g = g.expand(B, H, W, Z, G_, 3)

                # Reconstruct signals
                # predicted_f => [B, 1, H_, W_, Z_] => we pass it directly or squeeze
                signal_pred = fwdti_signal_prediction_lower_tri(
                    S0, f, b_scalar, Dg, D_iso_float, g
                ) * cropped_mask.permute(0, 2, 3, 4, 1) # B, H, W, Z, G

                # Compare with cropped_si => [B, G, H, W, Z] => reorder => [B, H, W, Z, G]
                cropped_si_ = cropped_si.permute(0, 2, 3, 4, 1)  # => [B, H, W, Z, G]
                loss_pinn = criterion(signal_pred, cropped_si_) * 0.0001

                # Final loss
                loss = loss_cf + loss_f  + loss_pinn + loss_pena
                pinn_loss_total += loss_pinn.item()

            else:
                loss = loss_cf + loss_f + loss_pena

        # Backprop with GradScaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        cf_loss_total += loss_cf.item()
        f_loss_total += loss_f.item()
        pena_loss_toal += loss_pena.item()


    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    elapsed_time = time.time() - start_time

    print(
        f"[Epoch {epoch}/{epochs}] - Loss: {avg_loss:.10} - Time: {elapsed_time:.2f}s - LR: {scheduler.get_last_lr()[0]:.10}")
    print(f"   CF: {cf_loss_total / len(train_loader):.10}, "
          f"F: {f_loss_total / len(train_loader):.10}, "
          f"PINN: {pinn_loss_total / len(train_loader):.10}"
          f"Pena: {pena_loss_toal / len(train_loader):.10}" )

    return avg_loss


def validate_one_epoch(
        model,
        val_loader,
        criterion,
        epoch,
        device,
        saved_fig_results="./saved_results"
):
    """
    Validation step on val_loader. Optionally saves the first batch's predictions to NIfTI.

    We assume each batch in val_loader returns the same shapes as train_loader,
    but we only compute CF & f loss (no PINN, or add if desired).
    """
    model.eval()
    val_loss = 0.0
    cf_loss_total = 0.0
    dti_loss_total = 0.0
    index = 0

    with torch.no_grad():
        for (
                cropped_sh,
                cropped_mask,
                cropped_cholesky_factors,
                cropped_f,
                cropped_dti,
                cropped_si,
                cropped_s0,
                bval,
                bvec
        ) in val_loader:
            # Move to GPU
            cropped_sh = cropped_sh.permute(0, 4, 1, 2, 3).float().to(device)  # [B, C, H, W, Z]
            cropped_mask = cropped_mask.permute(0, 4, 1, 2, 3).float().to(device)  # [B, 1, H, W, Z]
            cropped_cholesky_factors = cropped_cholesky_factors.permute(0, 4, 1, 2, 3).float().to(device)
            cropped_f = cropped_f.permute(0, 4, 1, 2, 3).float().to(device)  # [B, 1, H, W, Z]
            cropped_dti = cropped_dti.permute(0, 4, 1, 2, 3).float().to(device)  # optional

            # Forward
            predicted_cf_f = model(cropped_sh) * cropped_mask  # [B, 7, H, W, Z]
            predicted_cf = predicted_cf_f[:, :6]  # [B, 6, H, W, Z]
            predicted_f = predicted_cf_f[:, 6:]  # [B, 1, H, W, Z]

            # Mask GT
            cropped_cholesky_factors *= cropped_mask
            cropped_f *= cropped_mask
            cropped_dti *= cropped_mask  # optional

            # Loss
            # loss_f = criterion(predicted_f, cropped_f)
            cropped_cholesky_factors[:, 0] = torch.clamp(cropped_cholesky_factors[:, 0], max=0.05477225575051661, min=0)
            cropped_cholesky_factors[:, 1] = torch.clamp(cropped_cholesky_factors[:, 1], max=0.05477225575051661, min=0)
            cropped_cholesky_factors[:, 2] = torch.clamp(cropped_cholesky_factors[:, 2], max=0.05477225575051661, min=0)
            cropped_cholesky_factors[:, 3] = torch.clamp(cropped_cholesky_factors[:, 3], max=0.05, min=-0.05)
            cropped_cholesky_factors[:, 5] = torch.clamp(cropped_cholesky_factors[:, 5], max=0.05, min=-0.05)

            cropped_cholesky_factors[:, 4] = torch.clamp(cropped_cholesky_factors[:, 4], max=0.05, min=-0.05)


            loss_cf = criterion(predicted_cf, cropped_cholesky_factors)

            # predicted_transfered_dti = predicted_cf
            # Convert back to DTI
            # predicted_cf = torch.clamp(predicted_cf, max=0.05477225575051661, min=0)

            predicted_cf[:, 0] = torch.clamp(predicted_cf[:, 0], max=0.05477225575051661, min=0)
            predicted_cf[:, 1] = torch.clamp(predicted_cf[:, 1], max=0.05477225575051661, min=0)
            predicted_cf[:, 2] = torch.clamp(predicted_cf[:, 2], max=0.05477225575051661, min=0)
            predicted_cf[:, 3] = torch.clamp(predicted_cf[:, 3], max=0.05, min=-0.05)
            predicted_cf[:, 5] = torch.clamp(predicted_cf[:, 5], max=0.05, min=-0.05)

            predicted_cf[:, 4] = torch.clamp(predicted_cf[:, 4], max=0.05, min=-0.05)


            predicted_transfered_dti = cholesky_to_lower_triangular(predicted_cf.permute(0, 2, 3, 4, 1))

            loss_dti = criterion(predicted_transfered_dti, cropped_dti.permute(0, 2, 3, 4, 1))

            loss = loss_cf + loss_dti
            val_loss += loss.item()
            dti_loss_total += loss_dti.item()
            cf_loss_total += loss_cf.item()

            # Save the first batch as NIfTI for inspection
            if index == 0:
                out_dir = os.path.join(saved_fig_results, f"epoch_{epoch}")
                os.makedirs(out_dir, exist_ok=True)

                # Save predicted CF, predicted f, ground truths, etc.
                # predicted_cf => [B, 6, H, W, Z] => permute => [B, H, W, Z, 6] for saving

                save_tensor_to_nifti(
                    cropped_sh.permute(0, 2, 3, 4, 1),  # or you might want [B, 6, H, W, Z] directly
                    os.path.join(out_dir, "input")
                )

                save_tensor_to_nifti(
                    predicted_cf.permute(0, 2, 3, 4, 1),  # or you might want [B, 6, H, W, Z] directly
                    os.path.join(out_dir, "predicted_cf")
                )

                save_tensor_to_nifti(
                    predicted_f.permute(0, 2, 3, 4, 1),
                    os.path.join(out_dir, "predicted_f")
                )
                save_tensor_to_nifti(
                    cropped_cholesky_factors.permute(0, 2, 3, 4, 1),
                    os.path.join(out_dir, "gt_cholesky")
                )

                save_tensor_to_nifti(
                    cropped_f.permute(0, 2, 3, 4, 1),
                    os.path.join(out_dir, "gt_f")
                )
                save_tensor_to_nifti(
                    cropped_mask.permute(0, 2, 3, 4, 1),
                    os.path.join(out_dir, "mask")
                )
                save_tensor_to_nifti(
                    cropped_dti.permute(0, 2, 3, 4, 1),
                    os.path.join(out_dir, "gt_dti")
                )

                save_tensor_to_nifti(
                    predicted_transfered_dti,
                    os.path.join(out_dir, "predicted_dti")
                )

                index += 1

    avg_val_loss = val_loss / len(val_loader)
    avg_cf_loss = cf_loss_total / len(val_loader)
    avg_dti_loss = dti_loss_total / len((val_loader))
    print(f"[Validation] Epoch {epoch} - Avg Loss: {avg_val_loss:.10f}")
    print(f"[Validation] Epoch {epoch} - Avg CF Loss: {avg_cf_loss:.10f}")
    print(f"[Validation] Epoch {epoch} - Avg DTI Loss: {avg_dti_loss:.10f}")

    return avg_val_loss


##############################################################################
#                          5. Main Training Script
##############################################################################
def main():
    # ------------------------- Basic Config -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 321
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    # Hyperparameters
    epochs = 50
    batch_size = 16
    val_batch_size = 4
    lr = 1e-3
    weight_decay = 3e-5

    # Directories
    saved_model = "./saved_model_4"
    saved_fig = "./saved_fig_4"
    saved_fig_results = "./saved_results_4"
    os.makedirs(saved_model, exist_ok=True)
    os.makedirs(saved_fig, exist_ok=True)
    os.makedirs(saved_fig_results, exist_ok=True)

    # ------------------------- Datasets / Loaders -------------------------
    train_data_path = "../../neonatal_patches_tem_1.txt"
    val_data_path = "./neonatal_full_test.txt"  # separate validation file

    train_set = MedicalImageDataset_Patches(train_data_path)  # Should return shape [B, C, H, W, Z] per item
    val_set = MedicalImageDataset(val_data_path, crop_size=128, train=False)

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,  # Increase if CPU can handle it
        pin_memory=True
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ------------------------- Model / Loss / Opt / Scheduler -------------------------
    model = Conv3DModel(in_channels=28)

    model.load_state_dict(torch.load("./saved_model_3/current_model.pth"), strict=False)
    # model.load_state_dict(torch.load("./saved_model_0/"))
    model = model.to(device)

    criterion = nn.MSELoss(reduction = 'sum')
    criterion_pinn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision scaler
    scaler = GradScaler()

    # ------------------------- Tracking -------------------------
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # ------------------------- Training Loop -------------------------
    total_start = time.time()
    for epoch in range(1, epochs + 1):
        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            criterion_pinn = criterion_pinn,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            epochs=epochs,
            device=device
        )
        train_losses.append(train_loss)

        # Save "current" model
        torch.save(model.state_dict(), os.path.join(saved_model, "current_model.pth"))

        # Validate every 10 epochs
        if epoch % 10 == 0:
            val_loss = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
                saved_fig_results=saved_fig_results
            )
            val_losses.append(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(saved_model, "best_model.pth"))
                print("Saved best model")

    # ------------------------- Final Save and Plots -------------------------
    torch.save(model.state_dict(), os.path.join(saved_model, "final_model.pth"))
    total_time = time.time() - total_start
    print(f"Training complete. Total time: {total_time / 60:.2f} minutes.")

    # Plot losses
    if len(val_losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(val_losses, label='Validation Loss', color='blue')
        plt.title('Validation Loss')
        plt.xlabel('Validation Steps (# epochs / 10)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(saved_fig, 'validation_loss_plot.png'))
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(saved_fig, 'training_loss_plot.png'))
    plt.close()


if __name__ == "__main__":
    main()
