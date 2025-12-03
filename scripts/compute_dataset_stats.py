#!/usr/bin/env python3
"""
Compute comprehensive statistics from CityWalker dataset for DBR hyperparameter tuning.

This script analyzes:
1. Dataset file counts (videos, poses, depths)
2. Depth statistics (mean, std, min, max, percentiles)
3. Waypoint statistics (distances, angles, step scales)
4. Polar clearance statistics (with different DBR configs)
5. Frame dimensions and camera intrinsics
6. Pose statistics

Output: JSON file with all statistics and recommendations for DBR config.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.citywalk_dataset import CityWalkDataset
from data.citywalk_feat_dataset import CityWalkFeatDataset
from model.dbr import DepthPolarReducer


class DictNamespace:
    """Simple namespace for config"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DictNamespace(**cfg_dict)
    return cfg


def compute_depth_stats(depth_map, depth_mask=None):
    """Compute statistics for a single depth map"""
    if depth_mask is not None:
        valid_depths = depth_map[depth_mask]
    else:
        # Use valid depth range
        valid_mask = (depth_map > 0.1) & (depth_map < 100.0) & (~np.isnan(depth_map))
        valid_depths = depth_map[valid_mask]
    
    if len(valid_depths) == 0:
        return None
    
    stats = {
        'mean': float(np.mean(valid_depths)),
        'std': float(np.std(valid_depths)),
        'min': float(np.min(valid_depths)),
        'max': float(np.max(valid_depths)),
        'median': float(np.median(valid_depths)),
        'p10': float(np.percentile(valid_depths, 10)),
        'p25': float(np.percentile(valid_depths, 25)),
        'p75': float(np.percentile(valid_depths, 75)),
        'p90': float(np.percentile(valid_depths, 90)),
        'p95': float(np.percentile(valid_depths, 95)),
        'p99': float(np.percentile(valid_depths, 99)),
        'valid_ratio': float(np.sum(valid_depths > 0) / valid_depths.size) if valid_depths.size > 0 else 0.0,
    }
    return stats


def compute_waypoint_stats(waypoints, step_scale):
    """Compute statistics for waypoints"""
    # Convert to metric space
    waypoints_metric = waypoints * step_scale
    
    # Compute distances and angles
    deltas = np.diff(waypoints_metric, axis=0, prepend=waypoints_metric[0:1])
    distances = np.linalg.norm(deltas, axis=-1)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    
    stats = {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'median_distance': float(np.median(distances)),
        'mean_angle_deg': float(np.mean(np.abs(angles)) * 180 / np.pi),
        'max_angle_deg': float(np.max(np.abs(angles)) * 180 / np.pi),
        'step_scale': float(step_scale),
    }
    return stats


def compute_clearance_stats(depth_map, depth_mask, polar_reducer):
    """Compute clearance statistics using polar reducer"""
    depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)  # (1, H, W)
    mask_tensor = torch.from_numpy(depth_mask).bool().unsqueeze(0) if depth_mask is not None else None
    
    with torch.no_grad():
        clearance_vector, bin_centers = polar_reducer(depth_tensor, mask_tensor)
        clearance_vector = clearance_vector.squeeze(0).cpu().numpy()  # (num_bins,)
    
    stats = {
        'mean_clearance': float(np.mean(clearance_vector)),
        'std_clearance': float(np.std(clearance_vector)),
        'min_clearance': float(np.min(clearance_vector)),
        'max_clearance': float(np.max(clearance_vector)),
        'median_clearance': float(np.median(clearance_vector)),
        'p10_clearance': float(np.percentile(clearance_vector, 10)),
        'p25_clearance': float(np.percentile(clearance_vector, 25)),
        'p75_clearance': float(np.percentile(clearance_vector, 75)),
        'p90_clearance': float(np.percentile(clearance_vector, 90)),
    }
    return stats


def analyze_dataset(cfg, mode='train', max_samples=None, compute_clearance=True):
    """Analyze dataset and compute statistics"""
    print(f"\n{'='*60}")
    print(f"Analyzing {mode} dataset")
    print(f"{'='*60}\n")
    
    # Create dataset based on type
    dataset_type = getattr(cfg.data, 'type', 'citywalk')
    if dataset_type == 'citywalk_feat':
        dataset = CityWalkFeatDataset(cfg, mode=mode)
    else:
        dataset = CityWalkDataset(cfg, mode=mode)
    
    # File counts
    num_videos = len(dataset.video_path)
    num_poses = len(dataset.pose_path)
    # Handle depth paths - check if depth_path exists and count non-None entries
    if hasattr(dataset, 'depth_path') and dataset.depth_path is not None:
        num_depths = sum(1 for p in dataset.depth_path if p is not None)
    else:
        # For datasets without depth_path attribute or when depths are not loaded,
        # try to count from depth directory if it exists
        use_dbr = getattr(dataset, 'use_dbr', False) or getattr(dataset, 'load_depth_for_eval', False)
        if use_dbr and hasattr(dataset, 'depth_dir') and dataset.depth_dir:
            depth_dir = dataset.depth_dir
            if os.path.exists(depth_dir):
                # Count depth files recursively (for nested Frodobots structure)
                num_depths = len([f for root, dirs, files in os.walk(depth_dir) 
                                 for f in files if f.endswith('_depth.npy')])
            else:
                num_depths = 0
                print(f"Warning: Depth directory does not exist: {depth_dir}")
                print("  Skipping depth statistics (will still compute waypoint stats)")
        else:
            num_depths = 0
            if hasattr(dataset, 'load_depth_for_eval') and dataset.load_depth_for_eval:
                print("Note: Depth loading enabled but depth_dir not configured")
    
    print(f"Dataset files:")
    print(f"  Videos: {num_videos}")
    print(f"  Poses: {num_poses}")
    print(f"  Depths: {num_depths}")
    print(f"  Total samples: {len(dataset)}")
    
    # Collect statistics
    depth_stats_list = []
    waypoint_stats_list = []
    clearance_stats_list = []
    frame_dims = []
    step_scales = []
    
    # Initialize polar reducer if needed
    polar_reducer = None
    use_dbr = getattr(dataset, 'use_dbr', False) or getattr(dataset, 'load_depth_for_eval', False)
    if compute_clearance and use_dbr:
        # Try to get DBR config, use defaults if not available
        dbr_cfg = getattr(cfg.model, 'dbr', None)
        if dbr_cfg is None:
            print("Warning: DBR config not found, using default values for clearance computation")
            dbr_cfg = DictNamespace(
                num_bins=39,
                temperature=10.0,
                crop_bottom_ratio=0.6,
                fov_horizontal=90.0,
                fx=320.0,
                fy=320.0,
                cx=320.0,
                cy=180.0
            )
        intrinsics = {
            'fx': getattr(dbr_cfg, 'fx', 320.0),
            'fy': getattr(dbr_cfg, 'fy', 320.0),
            'cx': getattr(dbr_cfg, 'cx', 320.0),
            'cy': getattr(dbr_cfg, 'cy', 180.0),
        }
        polar_reducer = DepthPolarReducer(
            num_bins=getattr(dbr_cfg, 'num_bins', 39),
            temperature=getattr(dbr_cfg, 'temperature', 10.0),
            crop_bottom_ratio=getattr(dbr_cfg, 'crop_bottom_ratio', 0.6),
            fov_horizontal=getattr(dbr_cfg, 'fov_horizontal', 90.0),
            intrinsics=intrinsics
        )
        polar_reducer.eval()
    
    # Sample indices
    num_samples = len(dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    
    sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    print(f"\nProcessing {num_samples} samples...")
    
    for idx in tqdm(sample_indices, desc="Computing stats"):
        try:
            sample = dataset[idx]
            
            # Frame dimensions
            if 'video_frames' in sample:
                frames = sample['video_frames']
                if isinstance(frames, torch.Tensor):
                    # Shape: (N, C, H, W) where N is context_size
                    if frames.ndim == 4:
                        _, _, H, W = frames.shape
                    elif frames.ndim == 3:
                        _, H, W = frames.shape
                    else:
                        H, W = frames.shape[-2:]
                else:
                    # numpy array: (N, H, W, C) or (H, W, C)
                    if frames.ndim == 4:
                        _, H, W, _ = frames.shape
                    elif frames.ndim == 3:
                        H, W, _ = frames.shape
                    else:
                        H, W = frames.shape[:2]
                frame_dims.append((int(H), int(W)))
            
            # Step scale
            if 'step_scale' in sample:
                step_scales.append(sample['step_scale'].item() if isinstance(sample['step_scale'], torch.Tensor) else sample['step_scale'])
            
            # Depth statistics
            if 'depth_map' in sample and sample['depth_map'] is not None:
                depth_map = sample['depth_map']
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.numpy()
                
                depth_mask = None
                if 'depth_mask' in sample and sample['depth_mask'] is not None:
                    depth_mask = sample['depth_mask']
                    if isinstance(depth_mask, torch.Tensor):
                        depth_mask = depth_mask.numpy()
                
                depth_stats = compute_depth_stats(depth_map, depth_mask)
                if depth_stats is not None:
                    depth_stats_list.append(depth_stats)
                
                # Clearance statistics
                if compute_clearance and polar_reducer is not None:
                    clearance_stats = compute_clearance_stats(depth_map, depth_mask, polar_reducer)
                    clearance_stats_list.append(clearance_stats)
            
            # Waypoint statistics
            if 'waypoints' in sample and 'step_scale' in sample:
                waypoints = sample['waypoints']
                if isinstance(waypoints, torch.Tensor):
                    waypoints = waypoints.numpy()
                
                step_scale = sample['step_scale']
                if isinstance(step_scale, torch.Tensor):
                    step_scale = step_scale.item()
                
                waypoint_stats = compute_waypoint_stats(waypoints, step_scale)
                waypoint_stats_list.append(waypoint_stats)
        
        except Exception as e:
            print(f"\nWarning: Error processing sample {idx}: {e}")
            continue
    
    # Aggregate statistics
    results = {
        'file_counts': {
            'num_videos': num_videos,
            'num_poses': num_poses,
            'num_depths': num_depths,
            'num_samples': len(dataset),
            'num_processed': num_samples,
        }
    }
    
    # Frame dimensions
    if frame_dims:
        unique_dims = list(set(frame_dims))
        dim_counts = {str(dim): frame_dims.count(dim) for dim in unique_dims}
        results['frame_dimensions'] = {
            'unique_dimensions': unique_dims,
            'dimension_counts': dim_counts,
            'most_common': max(dim_counts.items(), key=lambda x: x[1])[0] if dim_counts else None,
        }
    
    # Depth statistics
    if depth_stats_list:
        depth_agg = {}
        for key in depth_stats_list[0].keys():
            values = [s[key] for s in depth_stats_list]
            depth_agg[f'{key}_mean'] = float(np.mean(values))
            depth_agg[f'{key}_std'] = float(np.std(values))
            depth_agg[f'{key}_min'] = float(np.min(values))
            depth_agg[f'{key}_max'] = float(np.max(values))
        results['depth_statistics'] = depth_agg
        results['depth_statistics']['num_samples'] = len(depth_stats_list)
    
    # Clearance statistics
    if clearance_stats_list:
        clearance_agg = {}
        for key in clearance_stats_list[0].keys():
            values = [s[key] for s in clearance_stats_list]
            clearance_agg[f'{key}_mean'] = float(np.mean(values))
            clearance_agg[f'{key}_std'] = float(np.std(values))
            clearance_agg[f'{key}_min'] = float(np.min(values))
            clearance_agg[f'{key}_max'] = float(np.max(values))
        results['clearance_statistics'] = clearance_agg
        results['clearance_statistics']['num_samples'] = len(clearance_stats_list)
    
    # Waypoint statistics
    if waypoint_stats_list:
        waypoint_agg = {}
        for key in waypoint_stats_list[0].keys():
            values = [s[key] for s in waypoint_stats_list]
            waypoint_agg[f'{key}_mean'] = float(np.mean(values))
            waypoint_agg[f'{key}_std'] = float(np.std(values))
            waypoint_agg[f'{key}_min'] = float(np.min(values))
            waypoint_agg[f'{key}_max'] = float(np.max(values))
        results['waypoint_statistics'] = waypoint_agg
        results['waypoint_statistics']['num_samples'] = len(waypoint_stats_list)
    
    # Step scale statistics
    if step_scales:
        results['step_scale_statistics'] = {
            'mean': float(np.mean(step_scales)),
            'std': float(np.std(step_scales)),
            'min': float(np.min(step_scales)),
            'max': float(np.max(step_scales)),
            'median': float(np.median(step_scales)),
            'p10': float(np.percentile(step_scales, 10)),
            'p90': float(np.percentile(step_scales, 90)),
        }
    
    return results


def generate_dbr_recommendations(stats):
    """Generate DBR configuration recommendations based on statistics"""
    recommendations = {}
    
    # Safety margin τ recommendation
    if 'clearance_statistics' in stats:
        clearance = stats['clearance_statistics']
        # Use p10 clearance as conservative safety margin
        if 'p10_clearance_mean' in clearance:
            recommendations['margin'] = {
                'recommended': max(0.3, clearance['p10_clearance_mean'] * 0.5),
                'conservative': max(0.5, clearance['p10_clearance_mean'] * 0.7),
                'aggressive': max(0.2, clearance['p10_clearance_mean'] * 0.3),
                'rationale': 'Based on p10 clearance to avoid collisions',
            }
    
    # Number of bins B recommendation
    if 'waypoint_statistics' in stats:
        waypoint = stats['waypoint_statistics']
        if 'max_angle_deg_mean' in waypoint:
            max_angle = waypoint['max_angle_deg_mean']
            # Recommend bins based on angular resolution needed
            # Want ~2-3 degree resolution
            recommended_bins = int(max_angle / 2.5)
            recommendations['num_bins'] = {
                'recommended': max(16, min(64, recommended_bins)),
                'rationale': f'Based on max waypoint angle ({max_angle:.1f} deg)',
            }
    
    # Temperature κ recommendation
    if 'depth_statistics' in stats:
        depth = stats['depth_statistics']
        if 'std_mean' in depth:
            # Higher temperature for sharper distributions
            std_val = depth['std_mean']
            recommendations['temperature'] = {
                'recommended': max(10.0, min(50.0, 20.0 * (1.0 / (std_val + 0.5)))),
                'rationale': f'Based on depth std ({std_val:.2f}m)',
            }
    
    # Crop bottom ratio recommendation
    if 'depth_statistics' in stats:
        recommendations['crop_bottom_ratio'] = {
            'recommended': 0.6,
            'range': [0.5, 0.7],
            'rationale': 'Focus on ground-level obstacles (50-70% of image)',
        }
    
    # Camera intrinsics recommendation
    if 'frame_dimensions' in stats:
        dims = stats['frame_dimensions']
        if 'most_common' in dims and dims['most_common']:
            H, W = eval(dims['most_common'])
            # Default intrinsics assuming typical camera
            recommendations['intrinsics'] = {
                'fx': float(W / 2),
                'fy': float(W / 2),
                'cx': float(W / 2),
                'cy': float(H / 2),
                'rationale': f'Based on frame dimensions {W}x{H}',
            }
    
    # Loss weight λ_bar recommendation
    recommendations['weight'] = {
        'recommended': 1.0,
        'range': [0.5, 2.0],
        'rationale': 'Start with 1.0, tune based on validation performance',
    }
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Compute dataset statistics for DBR')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset mode to analyze')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (None = all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: stats_{mode}.json)')
    parser.add_argument('--no-clearance', action='store_true',
                       help='Skip clearance statistics computation')
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Analyze dataset
    results = analyze_dataset(
        cfg,
        mode=args.mode,
        max_samples=args.max_samples,
        compute_clearance=not args.no_clearance
    )
    
    # Generate recommendations
    recommendations = generate_dbr_recommendations(results)
    results['dbr_recommendations'] = recommendations
    
    # Output file
    if args.output is None:
        args.output = f"stats_{args.mode}.json"
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Statistics saved to: {args.output}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("Summary:")
    print(f"  Processed {results['file_counts']['num_processed']} samples")
    if 'depth_statistics' in results:
        print(f"  Depth samples: {results['depth_statistics']['num_samples']}")
    if 'waypoint_statistics' in results:
        print(f"  Waypoint samples: {results['waypoint_statistics']['num_samples']}")
    if 'clearance_statistics' in results:
        print(f"  Clearance samples: {results['clearance_statistics']['num_samples']}")
    
    print("\nDBR Recommendations:")
    for key, value in recommendations.items():
        if isinstance(value, dict) and 'recommended' in value:
            print(f"  {key}: {value['recommended']}")
            if 'rationale' in value:
                print(f"    ({value['rationale']})")


if __name__ == '__main__':
    main()

