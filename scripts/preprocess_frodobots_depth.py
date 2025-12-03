#!/usr/bin/env python3
"""
Preprocess Depth Maps for Frodobots Dataset

This script finds all videos in the Frodobots nested directory structure
and processes them to generate depth maps.

Usage:
    python scripts/preprocess_frodobots_depth.py \
        --vast_path /vast/hs5580/data/frodobots_2k/extracted \
        --depth_dir /vast/hs5580/data/frodobots_2k/extracted/depths \
        --checkpoint <path_to_depth_model> \
        --model_size small
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from tqdm import tqdm
from model.depth_teacher import load_depth_teacher, precompute_depth_for_video


def find_frodobots_videos(vast_path):
    """
    Find all video files in Frodobots structure:
    part_X/output_rides_X/ride_XXXXX/recordings_converted_2min/*.mp4
    """
    video_files = []
    
    # Iterate through all part directories
    parts_dir = Path(vast_path)
    if not parts_dir.exists():
        print(f"Error: Path {vast_path} does not exist")
        return video_files
    
    for part_dir in sorted(parts_dir.glob("part_*")):
        if not part_dir.is_dir():
            continue
        
        part_num = part_dir.name.split("_")[1]
        output_rides_dir = part_dir / f"output_rides_{part_num}"
        
        if not output_rides_dir.exists():
            print(f"Warning: {output_rides_dir} not found")
            continue
        
        # Find all ride directories
        for ride_dir in sorted(output_rides_dir.glob("ride_*")):
            if not ride_dir.is_dir():
                continue
            
            recordings_dir = ride_dir / "recordings_converted_2min"
            if not recordings_dir.exists():
                continue
            
            # Find all video files in this recordings directory
            for video_file in sorted(recordings_dir.glob("*.mp4")):
                video_files.append({
                    'path': str(video_file),
                    'name': video_file.name,
                    'part': part_dir.name,
                    'ride': ride_dir.name
                })
    
    return video_files


def main():
    parser = argparse.ArgumentParser(description='Precompute depth maps for Frodobots dataset')
    parser.add_argument('--vast_path', type=str, required=True,
                       help='Root path to Frodobots extracted data')
    parser.add_argument('--depth_dir', type=str, required=True,
                       help='Directory to save depth maps')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to depth model checkpoint')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='Depth model size')
    parser.add_argument('--max_depth', type=float, default=20.0,
                       help='Maximum depth in meters')
    parser.add_argument('--save_format', type=str, default='npy',
                       choices=['npy', 'png16'],
                       help='Format to save depth maps')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing depth files')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    parser.add_argument('--video_fps', type=int, default=30,
                       help='Original video FPS')
    parser.add_argument('--target_fps', type=int, default=1,
                       help='Target FPS for depth extraction')
    
    args = parser.parse_args()
    
    # Create depth directory
    os.makedirs(args.depth_dir, exist_ok=True)
    print(f"Depth maps will be saved to {args.depth_dir}")
    
    # Find all videos
    print(f"\nScanning for videos in Frodobots structure...")
    video_files = find_frodobots_videos(args.vast_path)
    print(f"Found {len(video_files)} videos to process")
    
    if len(video_files) == 0:
        print("No videos found! Check the vast_path.")
        return
    
    # Load depth teacher model
    print(f"\nLoading Depth-Anything-V2 model ({args.model_size})...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    depth_teacher = load_depth_teacher(
        model_size=args.model_size,
        max_depth=args.max_depth,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    print("Model loaded successfully!")
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for video_info in tqdm(video_files, desc="Processing videos"):
        video_path = video_info['path']
        video_name = video_info['name']
        output_name = video_name.replace('.mp4', '_depth.npy')
        output_path = os.path.join(args.depth_dir, output_name)
        
        # Skip if already exists and not forcing
        if os.path.exists(output_path) and not args.force:
            skipped += 1
            continue
        
        try:
            # Precompute depth for this video
            precompute_depth_for_video(
                video_path=video_path,
                output_path=output_path,
                depth_teacher=depth_teacher,
                target_fps=args.target_fps,
                video_fps=args.video_fps,
                save_format=args.save_format
            )
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {video_name}: {e}")
            failed += 1
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Frodobots Depth Preprocessing Summary")
    print("="*60)
    print(f"Total videos:     {len(video_files)}")
    print(f"Processed:       {processed}")
    print(f"Skipped:          {skipped}")
    print(f"Failed:           {failed}")
    print("="*60)
    
    if failed > 0:
        print("\nSome videos failed to process. Check the error messages above.")
    else:
        print("\nAll videos processed successfully!")
        print(f"\nDepth maps saved to: {args.depth_dir}")


if __name__ == '__main__':
    main()

