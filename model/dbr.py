"""
Depth Barrier Regularization (DBR) Module

This module implements depth-based safety regularization for waypoint prediction.
It computes a polar clearance vector from monocular depth and penalizes predicted 
waypoint directions whose forward clearance falls below a safety margin.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DepthPolarReducer(nn.Module):
    """
    Reduces depth maps to polar clearance vectors using soft-min over yaw bins.
    
    Args:
        num_bins (int): Number of yaw bins (B in the paper)
        temperature (float): Temperature for soft-min (κ in the paper)
        crop_bottom_ratio (float): Ratio of bottom rows to keep (0.5-0.6)
        fov_horizontal (float): Horizontal field of view in degrees
        intrinsics (dict): Camera intrinsics with keys 'fx', 'fy', 'cx', 'cy'
    """
    
    def __init__(self, num_bins=32, temperature=20.0, crop_bottom_ratio=0.6, 
                 fov_horizontal=90.0, intrinsics=None):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        self.crop_bottom_ratio = crop_bottom_ratio
        self.fov_horizontal = fov_horizontal
        
        # Default intrinsics for typical camera (will be overridden if provided)
        if intrinsics is None:
            intrinsics = {'fx': 320.0, 'fy': 320.0, 'cx': 320.0, 'cy': 180.0}
        
        self.register_buffer('fx', torch.tensor(intrinsics['fx']))
        self.register_buffer('fy', torch.tensor(intrinsics['fy']))
        self.register_buffer('cx', torch.tensor(intrinsics['cx']))
        self.register_buffer('cy', torch.tensor(intrinsics['cy']))
        
        # Precompute bin centers
        fov_rad = np.deg2rad(fov_horizontal)
        bin_centers = torch.linspace(-fov_rad / 2, fov_rad / 2, num_bins)
        self.register_buffer('bin_centers', bin_centers)
        
        # Bin width
        self.bin_width = fov_rad / num_bins
    
    def compute_per_pixel_yaw(self, H, W):
        """
        Compute per-pixel yaw angles α(u,v) = arctan2((u-cx)/fx, 1)
        Uses stored intrinsics (for backward compatibility).
        
        Args:
            H, W: Height and width of the depth map
            
        Returns:
            yaw_map: (H, W) tensor of yaw angles
        """
        return self._compute_per_pixel_yaw_with_intrinsics(H, W, self.fx, self.fy, self.cx, self.cy)
    
    def compute_triangular_weights(self, yaw_map):
        """
        Compute triangular bin membership weights for each pixel.
        
        Args:
            yaw_map: (H, W) tensor of yaw angles
            
        Returns:
            weights: (B, H, W) tensor of normalized bin weights
        """
        # Compute distance from each pixel to each bin center
        # yaw_map: (H, W), bin_centers: (B,)
        # diff: (B, H, W)
        diff = torch.abs(yaw_map.unsqueeze(0) - self.bin_centers.view(-1, 1, 1))
        
        # Triangular weighting: max(0, 1 - |α - ψ_b| / Δ)
        weights = torch.clamp(1.0 - diff / self.bin_width, min=0.0)
        
        # Normalize weights so they sum to 1 across bins for each pixel
        weight_sum = weights.sum(dim=0, keepdim=True) + 1e-8
        weights = weights / weight_sum
        
        return weights  # (B, H, W)
    
    def soft_min_depth(self, depth_map, weights, mask=None):
        """
        Compute soft-min depth per bin using temperature-based weighting.
        
        Args:
            depth_map: (B, H, W) depth values
            weights: (num_bins, H, W) triangular bin weights
            mask: (B, H, W) optional mask for invalid pixels
            
        Returns:
            min_depths: (B, num_bins) soft-min depth per bin for each sample
        """
        B, H, W = depth_map.shape
        
        # Expand depth to (B, num_bins, H, W)
        depth_expanded = depth_map.unsqueeze(1).expand(-1, self.num_bins, -1, -1)
        
        # Expand weights to (1, num_bins, H, W) then broadcast
        weights_expanded = weights.unsqueeze(0)  # (1, num_bins, H, W)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has the same spatial dimensions as depth_map
            assert mask.shape == depth_map.shape, f"Mask shape {mask.shape} doesn't match depth shape {depth_map.shape}"
            mask_expanded = mask.unsqueeze(1).expand(-1, self.num_bins, -1, -1)
            weights_expanded = weights_expanded * mask_expanded.float()
        
        # Soft-min: r_b = -1/κ * log(sum_pixels(exp(-κ * D) * w_b))
        exp_term = torch.exp(-self.temperature * depth_expanded)
        weighted_exp = exp_term * weights_expanded
        
        # Sum over spatial dimensions
        sum_exp = weighted_exp.sum(dim=(2, 3)) + 1e-8  # (B, num_bins)
        
        # Compute soft-min
        min_depths = -torch.log(sum_exp) / self.temperature
        
        return min_depths  # (B, num_bins)
    
    def forward(self, depth_map, mask=None, resize_ratio_h=None, resize_ratio_w=None):
        """
        Convert depth map to polar clearance vector.
        
        Args:
            depth_map: (B, H, W) depth values in meters
            mask: (B, H, W) optional boolean mask (True for valid pixels)
            resize_ratio_h: Optional height resize ratio (new_h / orig_h) for intrinsics adjustment
            resize_ratio_w: Optional width resize ratio (new_w / orig_w) for intrinsics adjustment
            
        Returns:
            clearance_vector: (B, num_bins) min depth per yaw bin
            bin_centers: (num_bins,) yaw angle for each bin center
        """
        B, H, W = depth_map.shape
        
        # Adjust intrinsics if depth was resized
        # When depth is resized, camera intrinsics need to be scaled proportionally
        if resize_ratio_h is not None and resize_ratio_w is not None:
            fx_adjusted = self.fx * resize_ratio_w
            fy_adjusted = self.fy * resize_ratio_h
            cx_adjusted = self.cx * resize_ratio_w
            cy_adjusted = self.cy * resize_ratio_h
        else:
            # Use original intrinsics (assume depth matches intrinsics resolution)
            fx_adjusted = self.fx
            fy_adjusted = self.fy
            cx_adjusted = self.cx
            cy_adjusted = self.cy
        
        # Crop to bottom portion (focus on ground-level obstacles)
        crop_start = int(H * (1.0 - self.crop_bottom_ratio))
        depth_cropped = depth_map[:, crop_start:, :]
        if mask is not None:
            mask_cropped = mask[:, crop_start:, :]
        else:
            mask_cropped = None
        
        # Adjust cy for cropped region
        cy_crop_adjusted = cy_adjusted - crop_start
        
        # Compute per-pixel yaw for cropped region using adjusted intrinsics
        H_crop = depth_cropped.shape[1]
        yaw_map = self._compute_per_pixel_yaw_with_intrinsics(
            H_crop, W, fx_adjusted, fy_adjusted, cx_adjusted, cy_crop_adjusted
        )  # (H_crop, W)
        
        # Compute triangular weights
        weights = self.compute_triangular_weights(yaw_map)  # (num_bins, H_crop, W)
        
        # Compute soft-min depth per bin
        clearance_vector = self.soft_min_depth(depth_cropped, weights, mask_cropped)
        
        return clearance_vector, self.bin_centers
    
    def _compute_per_pixel_yaw_with_intrinsics(self, H, W, fx, fy, cx, cy):
        """
        Compute per-pixel yaw angles with given intrinsics.
        
        Args:
            H, W: Height and width of the depth map
            fx, fy, cx, cy: Camera intrinsics
            
        Returns:
            yaw_map: (H, W) tensor of yaw angles
        """
        # Create pixel coordinate grid
        v = torch.arange(H, device=fy.device).float()  # height indices
        u = torch.arange(W, device=fx.device).float()  # width indices
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')  # (H, W) grids
        
        # Compute yaw per pixel: α(u,v) = arctan2((u-cx)/fx, 1)
        yaw_map = torch.atan2((u_grid - cx) / fx, torch.ones_like(u_grid))
        
        return yaw_map  # (H, W)


class BarrierLoss(nn.Module):
    """
    Barrier loss that penalizes waypoints whose direction has insufficient clearance.
    
    Args:
        margin (float): Safety margin τ in meters
        weight (float): Loss weight λ_bar
    """
    
    def __init__(self, margin=0.5, weight=1.0):
        super().__init__()
        self.margin = margin
        self.weight = weight
    
    def bilinear_interpolate(self, clearance_vector, bin_centers, yaw_angles):
        """
        Differentiable interpolation of clearance values at given yaw angles.
        Uses soft attention weights instead of hard indexing to preserve gradients.
        
        Args:
            clearance_vector: (B, num_bins) clearance values
            bin_centers: (num_bins,) bin center yaw angles
            yaw_angles: (B, T) query yaw angles
            
        Returns:
            interpolated: (B, T) interpolated clearance values
        """
        B, T = yaw_angles.shape
        num_bins = bin_centers.shape[0]
        
        # Expand for broadcasting
        bin_centers_exp = bin_centers.view(1, 1, -1)  # (1, 1, num_bins)
        yaw_exp = yaw_angles.unsqueeze(-1)  # (B, T, 1)
        
        # Compute distances to all bins
        diffs = yaw_exp - bin_centers_exp  # (B, T, num_bins)
        distances = torch.abs(diffs)
        
        # Use soft attention weights (differentiable)
        # Smaller distance = higher weight
        # Use negative distance with softmax for smooth interpolation
        temperature = 0.1  # Controls sharpness of interpolation
        weights = F.softmax(-distances / temperature, dim=-1)  # (B, T, num_bins)
        
        # Weighted sum of clearance values (differentiable)
        clearance_exp = clearance_vector.unsqueeze(1)  # (B, 1, num_bins)
        interpolated = (weights * clearance_exp).sum(dim=-1)  # (B, T)
        
        return interpolated  # (B, T)
    
    def forward(self, waypoints, clearance_vector, bin_centers):
        """
        Compute barrier loss for predicted waypoints.
        
        Args:
            waypoints: (B, T, 2) predicted waypoints in ego frame (x, y)
            clearance_vector: (B, num_bins) min depth per yaw bin
            bin_centers: (num_bins,) yaw angles for bins
            
        Returns:
            loss: scalar barrier loss
        """
        B, T, _ = waypoints.shape
        
        # Compute yaw angle for each waypoint
        # φ_t = atan2(y_t, x_t)
        yaw_angles = torch.atan2(waypoints[:, :, 1], waypoints[:, :, 0])  # (B, T)
        
        # Interpolate clearance at each waypoint direction
        d_min = self.bilinear_interpolate(clearance_vector, bin_centers, yaw_angles)  # (B, T)
        
        # Compute barrier loss: softplus(τ - d_min)
        barrier = F.softplus(self.margin - d_min)
        
        # Average over waypoints and batch
        loss = barrier.mean()
        
        return loss * self.weight


class DBRModule(nn.Module):
    """
    Complete Depth Barrier Regularization module.
    
    Combines polar reducer and barrier loss for end-to-end training.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Extract DBR config
        dbr_cfg = cfg.model.dbr
        
        # Initialize polar reducer
        intrinsics = {
            'fx': getattr(dbr_cfg, 'fx', 320.0),
            'fy': getattr(dbr_cfg, 'fy', 320.0),
            'cx': getattr(dbr_cfg, 'cx', 320.0),
            'cy': getattr(dbr_cfg, 'cy', 180.0),
        }
        
        self.polar_reducer = DepthPolarReducer(
            num_bins=dbr_cfg.num_bins,
            temperature=dbr_cfg.temperature,
            crop_bottom_ratio=dbr_cfg.crop_bottom_ratio,
            fov_horizontal=dbr_cfg.fov_horizontal,
            intrinsics=intrinsics
        )
        
        # Initialize barrier loss
        self.barrier_loss = BarrierLoss(
            margin=dbr_cfg.margin,
            weight=dbr_cfg.weight
        )
    
    def forward(self, waypoints, depth_map, depth_mask=None, resize_ratio_h=None, resize_ratio_w=None):
        """
        Compute DBR loss for predicted waypoints given depth map.
        
        Args:
            waypoints: (B, T, 2) predicted waypoints in ego frame
            depth_map: (B, H, W) depth values in meters
            depth_mask: (B, H, W) optional mask for valid depth pixels
            resize_ratio_h: Optional height resize ratio for intrinsics adjustment
            resize_ratio_w: Optional width resize ratio for intrinsics adjustment
            
        Returns:
            loss: scalar DBR loss
            clearance_vector: (B, num_bins) for visualization/logging
        """
        # Convert depth to polar clearance with adjusted intrinsics
        clearance_vector, bin_centers = self.polar_reducer(
            depth_map, depth_mask, resize_ratio_h=resize_ratio_h, resize_ratio_w=resize_ratio_w
        )
        
        # Compute barrier loss
        loss = self.barrier_loss(waypoints, clearance_vector, bin_centers)
        
        return loss, clearance_vector

