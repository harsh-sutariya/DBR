import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF
from decord import VideoReader, cpu
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random

class CityWalkSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = self.create_indices()

    def create_indices(self):
        indices = []
        for start_idx, end_idx in self.dataset.video_ranges:
            video_indices = list(range(start_idx, end_idx))
            if self.dataset.mode == 'train':
                random.shuffle(video_indices)
            indices.extend(video_indices)
        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class CityWalkFeatDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        
        # Construct full paths using vast_path + relative paths
        vast_path = getattr(cfg.data, 'vast_path', '')
        frodobots_structure = getattr(cfg.data, 'frodobots_structure', False)
        
        if frodobots_structure and vast_path:
            # Special handling for frodobots complex directory structure
            # Videos are in: part_X/output_rides_X/ride_XXXXX/recordings_converted_2min/
            # Poses are in: part_X/ride_XXXXX/dpvo_poses/
            self.video_dir = vast_path  # We'll scan subdirectories
            self.pose_dir = vast_path   # We'll scan subdirectories
            self.frodobots_mode = True
        elif vast_path:
            self.video_dir = os.path.join(vast_path, cfg.data.video_dir)
            self.pose_dir = os.path.join(vast_path, cfg.data.pose_dir)
            self.frodobots_mode = False
        else:
            # Fallback to old behavior for backward compatibility
            self.video_dir = cfg.data.video_dir
            self.pose_dir = cfg.data.pose_dir
            self.frodobots_mode = False
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.video_fps = cfg.data.video_fps
        self.pose_fps = cfg.data.pose_fps
        self.target_fps = cfg.data.target_fps
        self.num_workers = cfg.data.num_workers
        self.input_noise = cfg.data.input_noise
        self.search_window = cfg.data.search_window
        self.arrived_threshold = cfg.data.arrived_threshold
        self.arrived_prob = cfg.data.arrived_prob

        # DBR (Depth Barrier Regularization) support
        self.use_dbr = getattr(cfg.model, 'use_dbr', False)
        if self.use_dbr:
            depth_dir_config = getattr(cfg.data, 'depth_dir', None)
            if depth_dir_config and vast_path:
                self.depth_dir = os.path.join(vast_path, depth_dir_config)
            else:
                # Fallback to absolute path or original config
                self.depth_dir = depth_dir_config
            self.depth_mode = getattr(cfg.data, 'depth_mode', 'precomputed')  # 'precomputed' or 'online'
            
            if self.depth_mode == 'precomputed':
                if self.depth_dir is None:
                    raise ValueError("DBR is enabled with precomputed mode but depth_dir is not specified")
                self.depth_cache_mode = getattr(cfg.data, 'depth_cache_mode', 'on_the_fly')
                self.depth_cache = {}  # Cache for preloaded depth data
            elif self.depth_mode == 'online':
                # Load depth model for online inference
                print("Loading depth model for online inference...")
                from model.depth_teacher import load_depth_teacher
                depth_checkpoint = getattr(cfg.data, 'depth_checkpoint', None)
                if depth_checkpoint is None:
                    raise ValueError("DBR online mode requires depth_checkpoint in config")
                self.depth_model = load_depth_teacher(
                    model_size=getattr(cfg.data, 'depth_model_size', 'small'),
                    max_depth=getattr(cfg.data, 'depth_max_depth', 20.0),
                    checkpoint_path=depth_checkpoint,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print(f"Depth model loaded successfully ({cfg.data.depth_model_size})")
            else:
                raise ValueError(f"Invalid depth_mode: {self.depth_mode}. Use 'precomputed' or 'online'")
        else:
            self.depth_dir = None
            self.depth_mode = None

        # Load pose paths
        if self.frodobots_mode:
            # For frodobots: scan part_X/ride_XXXXX/dpvo_poses/ directories
            self.pose_path = []
            for part_dir in sorted(os.listdir(self.pose_dir)):
                if part_dir.startswith('part_'):
                    part_path = os.path.join(self.pose_dir, part_dir)
                    if os.path.isdir(part_path):
                        for ride_dir in sorted(os.listdir(part_path)):
                            if ride_dir.startswith('ride_'):
                                dpvo_poses_path = os.path.join(part_path, ride_dir, 'dpvo_poses')
                                if os.path.isdir(dpvo_poses_path):
                                    for f in sorted(os.listdir(dpvo_poses_path)):
                                        if f.endswith('.txt'):
                                            self.pose_path.append(os.path.join(dpvo_poses_path, f))
        else:
            # Normal structure: flat directory with pose files
            self.pose_path = [
                os.path.join(self.pose_dir, f)
                for f in sorted(os.listdir(self.pose_dir))
                if f.endswith('.txt')
            ]
        print(len(self.pose_path))
        if mode == 'train':
            self.pose_path = self.pose_path[:cfg.data.num_train]
        elif mode == 'val':
                        self.pose_path = self.pose_path[-cfg.data.num_val:]
        elif mode == 'test':
            self.pose_path = self.pose_path[-cfg.data.num_test:]
        else:
            raise ValueError(f"Invalid mode {mode}")

        # Load corresponding video paths
        self.video_path = []
        for f in self.pose_path:
            video_file = os.path.basename(f).replace(".txt", ".mp4")
            
            if self.frodobots_mode:
                # For frodobots: pose is in part_X/ride_XXXXX/dpvo_poses/
                # Video is in part_X/output_rides_X/ride_XXXXX/recordings_converted_2min/
                # Extract part number and ride directory from pose path
                pose_parts = f.split(os.sep)
                for i, part in enumerate(pose_parts):
                    if part.startswith('part_'):
                        part_num = part.split('_')[1]  # Extract X from part_X
                        ride_dir = pose_parts[i + 1]   # Get ride_XXXXX directory
                        video_path = os.path.join(
                            self.video_dir, 
                            f'part_{part_num}', 
                            f'output_rides_{part_num}', 
                            ride_dir, 
                            'recordings_converted_2min', 
                            video_file
                        )
                        break
                else:
                    raise ValueError(f"Could not parse frodobots path structure from {f}")
                video = video_path
            else:
                # Normal structure: video in same directory structure as pose
                video = os.path.join(self.video_dir, video_file)
                
            if not os.path.exists(video):
                raise FileNotFoundError(f"Video file {video} does not exist.")
            self.video_path.append(video)

        # Load corresponding depth paths if DBR is enabled
        if self.use_dbr and self.depth_mode == 'precomputed':
            self.depth_path = []
            for f in self.pose_path:
                depth_file = os.path.basename(f).replace(".txt", "_depth.npy")
                depth_path = os.path.join(self.depth_dir, depth_file)
                if os.path.exists(depth_path):
                    self.depth_path.append(depth_path)
                else:
                    print(f"Warning: Depth file {depth_path} does not exist. Setting to None.")
                    self.depth_path.append(None)
                    
            # Preload depth data if cache mode is 'preload'
            if self.depth_cache_mode == 'preload':
                print("Preloading depth data...")
                for video_idx, depth_path in enumerate(tqdm(self.depth_path, desc="Loading depth files")):
                    if depth_path is not None:
                        try:
                            depth_data = np.load(depth_path)  # (T, H, W)
                            self.depth_cache[video_idx] = depth_data
                        except Exception as e:
                            print(f"Warning: Failed to preload depth file {depth_path}: {e}")
                            self.depth_cache[video_idx] = None
                    else:
                        self.depth_cache[video_idx] = None
                print(f"Preloaded depth data for {len(self.depth_cache)} videos")
        else:
            self.depth_path = None

        # Load poses and compute usable counts
        self.poses = []
        self.count = []
        for f in tqdm(self.pose_path, desc="Loading poses"):
            pose_data = np.loadtxt(f)
            # Handle 1D arrays (single line files) by reshaping to 2D
            if pose_data.ndim == 1:
                pose_data = pose_data.reshape(1, -1)
            pose = pose_data[::max(1, self.pose_fps // self.target_fps), 1:]
            pose_nan = np.isnan(pose).any(axis=1)
            if np.any(pose_nan):
                first_nan_idx = np.argmin(pose_nan)
                pose = pose[:first_nan_idx]
            self.poses.append(pose)
            usable = pose.shape[0] - self.context_size - max(self.arrived_threshold*2, self.wp_length)
            self.count.append(max(usable, 0))  # Ensure non-negative

        # Remove pose files with zero usable samples
        valid_indices = [i for i, c in enumerate(self.count) if c > 0]
        self.poses = [self.poses[i] for i in valid_indices]
        self.video_path = [self.video_path[i] for i in valid_indices]
        self.count = [self.count[i] for i in valid_indices]
        self.step_scale = []
        for pose in self.poses:
            step_scale = np.linalg.norm(np.diff(pose[:, [0, 2]], axis=0), axis=1).mean()
            self.step_scale.append(step_scale)

        # Build the look-up table and video_ranges
        self.lut = []
        self.video_ranges = []
        idx_counter = 0
        for video_idx, count in enumerate(self.count):
            start_idx = idx_counter
            interval = self.context_size
            for pose_start in range(0, count, interval):
                self.lut.append((video_idx, pose_start))
                idx_counter += 1
            end_idx = idx_counter
            self.video_ranges.append((start_idx, end_idx))
        assert len(self.lut) > 0, "No usable samples found."

        # Initialize the video reader cache per worker
        self.video_reader_cache = {'video_idx': None, 'video_reader': None}

    def load_depth_frames(self, video_idx, pose_start, last_frame_tensor=None):
        """
        Load depth data for the last observation frame.
        
        Args:
            video_idx: Index of the video
            pose_start: Starting pose index
            last_frame_tensor: Last RGB frame tensor (for online depth inference)
            
        Returns:
            depth_map: (H, W) depth values in meters
            depth_mask: (H, W) boolean mask for valid depth pixels
        """
        if not self.use_dbr:
            return None, None
            
        if self.depth_mode == 'online':
            # Online depth inference
            if last_frame_tensor is None:
                print("Warning: Online depth mode requires last_frame_tensor")
                return None, None
            
            try:
                # Ensure tensor is on the right device and format
                if last_frame_tensor.dim() == 3:  # (C, H, W)
                    last_frame_tensor = last_frame_tensor.unsqueeze(0)  # (1, C, H, W)
                
                # Predict depth
                with torch.no_grad():
                    depth_map = self.depth_model.infer_batch_with_padding(last_frame_tensor)
                    depth_map = depth_map.squeeze(0).cpu().numpy()  # (H, W)
                
                # Create mask for valid depth values
                depth_mask = (depth_map > 0.1) & (depth_map < 100.0) & (~np.isnan(depth_map))
                
                return depth_map, depth_mask
                
            except Exception as e:
                print(f"Warning: Online depth inference failed: {e}")
                return None, None
        
        elif self.depth_mode == 'precomputed':
            # Load precomputed depth from disk
            if self.depth_path is None or self.depth_path[video_idx] is None:
                return None, None
            
            # Load depth data (on-the-fly or from cache)
            if video_idx in self.depth_cache:
                # Use cached data
                depth_data = self.depth_cache[video_idx]
                if depth_data is None:
                    return None, None
            elif self.depth_cache_mode == 'preload':
                # Preload mode but not in cache - data should have been preloaded
                return None, None
            else:
                # Load on-the-fly
                try:
                    depth_data = np.load(self.depth_path[video_idx])  # (T, H, W)
                except Exception as e:
                    print(f"Warning: Failed to load depth file {self.depth_path[video_idx]}: {e}")
                    return None, None
            
            # Get the depth frame corresponding to the last observation frame
            # The last frame index is at pose_start + context_size - 1
            depth_frame_idx = pose_start + self.context_size - 1
            
            # Ensure index is within bounds
            if depth_frame_idx >= depth_data.shape[0]:
                depth_frame_idx = depth_data.shape[0] - 1
            
            depth_map = depth_data[depth_frame_idx]  # (H, W)
            
            # Create mask for valid depth values (non-zero, non-nan)
            depth_mask = (depth_map > 0.1) & (depth_map < 100.0) & (~np.isnan(depth_map))
            
            return depth_map, depth_mask
        
        return None, None

    def __len__(self):
        return len(self.lut)

    def __getitem__(self, index):
        video_idx, pose_start = self.lut[index]

        # Retrieve or create the VideoReader for the current video
        if self.video_reader_cache['video_idx'] != video_idx:
            # Replace the old VideoReader with the new one
            self.video_reader_cache['video_reader'] = VideoReader(self.video_path[video_idx], ctx=cpu(0))
            self.video_reader_cache['video_idx'] = video_idx
        video_reader = self.video_reader_cache['video_reader']

        frame_multiplier = self.video_fps // self.target_fps
        start_frame_idx = pose_start * frame_multiplier
        frame_indices = start_frame_idx + np.arange(self.context_size + self.wp_length) * frame_multiplier

        # Ensure frame indices are within the video length
        num_frames = len(video_reader)
        frame_indices = [min(idx, num_frames - 1) for idx in frame_indices]

        # Load the required frames
        frames = video_reader.get_batch(frame_indices).asnumpy()

        # Process frames
        frames = self.process_frames(frames)
        input_frames = frames[:self.context_size]
        target_frames = frames[self.context_size:]

        # Get pose data
        pose = self.poses[video_idx]

        # Get input and future poses
        input_poses, future_poses = self.get_input_and_future_poses(pose, pose_start)
        original_input_poses = np.copy(input_poses)  # Store original poses before noise

        # Select target pose
        target_pose, arrived = self.select_target_pose(future_poses)

        # Determine arrived label
        # arrived = self.determine_arrived_label(input_poses[-1, :3], target_pose[:3])

        # Extract waypoints
        waypoint_poses = self.extract_waypoints(pose, pose_start)

        # # Add noise if necessary
        # if self.input_noise > 0:
        #     input_poses = self.add_noise(input_poses)

        # Transform poses
        current_pose = input_poses[-1]
        if self.cfg.model.cord_embedding.type == 'polar':
            transformed_input_positions = self.input2target(input_poses, target_pose)
        elif self.cfg.model.cord_embedding.type == 'input_target':
            transformed_input_positions = np.concatenate([
                self.transform_poses(input_poses, current_pose)[:, [0, 2]], 
                self.transform_target_pose(target_pose, current_pose)[np.newaxis, [0, 2]]
            ], axis=0)
        else:
            raise NotImplementedError(f"Coordinate embedding type {self.cfg.model.cord_embedding} not implemented")
        waypoints_transformed = self.transform_waypoints(waypoint_poses, current_pose)

        # Convert data to tensors - ensure all tensors are independent copies
        input_positions = torch.tensor(transformed_input_positions, dtype=torch.float32).clone()
        waypoints_transformed = torch.tensor(waypoints_transformed[:, [0, 2]], dtype=torch.float32).clone()
        step_scale = torch.tensor(self.step_scale[video_idx], dtype=torch.float32).clone()
        step_scale = torch.clamp(step_scale, min=1e-2)
        input_positions_scaled = (input_positions / step_scale).clone()
        waypoints_scaled = (waypoints_transformed / step_scale).clone()
        input_positions_scaled[:self.context_size-1] += torch.randn(self.context_size-1, 2) * self.input_noise
        arrived = torch.tensor(arrived, dtype=torch.float32).clone()
        
        # Ensure video frames are contiguous and independent
        input_frames = input_frames.contiguous().clone()
        target_frames = target_frames.contiguous().clone()
        
        sample = {
            'video_frames': input_frames,
            'future_video_frames': target_frames,
            'input_positions': input_positions_scaled,
            'waypoints': waypoints_scaled,
            'arrived': arrived,
            'step_scale': step_scale
        }

        # Add depth data if DBR is enabled
        if self.use_dbr:
            # Load depth data for the last observation frame
            last_frame_tensor = input_frames[-1] if self.depth_mode == 'online' else None
            depth_map, depth_mask = self.load_depth_frames(video_idx, pose_start, last_frame_tensor)
            
            if depth_map is not None and depth_mask is not None:
                # Ensure consistent shape - depth maps should be 2D (H, W)
                if depth_map.ndim != 2 or depth_mask.ndim != 2:
                    print(f"Warning: Invalid depth shape - depth_map: {depth_map.shape}, depth_mask: {depth_mask.shape}")
                    # Skip depth for this sample
                else:
                    # Ensure tensors are contiguous and don't share memory to avoid collation issues
                    sample['depth_map'] = torch.from_numpy(np.ascontiguousarray(depth_map)).float().clone()
                    sample['depth_mask'] = torch.from_numpy(np.ascontiguousarray(depth_mask)).bool().clone()

        # For visualization during validation
        if self.mode in ['val', 'test']:
            # vis_input_positions = self.transform_poses(input_poses, current_pose)
            transformed_original_input_positions = self.transform_poses(original_input_poses, current_pose)
            target_transformed = self.transform_target_pose(target_pose, current_pose)

            original_input_positions = torch.tensor(transformed_original_input_positions[:, [0, 2]], dtype=torch.float32).clone()
            # noisy_input_positions = torch.tensor(vis_input_positions[:, [0, 2]], dtype=torch.float32)
            noisy_input_positions = (input_positions_scaled[:-1] * step_scale).clone()
            target_transformed_position = torch.tensor(target_transformed[[0, 2]], dtype=torch.float32).clone()  # Only X and Z
            sample['original_input_positions'] = original_input_positions
            sample['noisy_input_positions'] = noisy_input_positions
            sample['gt_waypoints'] = waypoints_transformed.clone()
            sample['target_transformed'] = target_transformed_position  # Add target coordinate
        return sample

    def transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]
        return positions

    def get_input_and_future_poses(self, pose, pose_start):
        input_poses = pose[pose_start: pose_start + self.context_size]
        search_end = min(pose_start + self.context_size + self.search_window, pose.shape[0])
        future_poses = pose[pose_start + self.context_size: search_end]
        if future_poses.shape[0] == 0:
            raise IndexError(f"No future poses available for index {pose_start}.")
        return input_poses, future_poses
    
    def input2target(self, input_poses, target_pose):
        input_positions = input_poses[:, :3]
        target_position = target_pose[:3]
        transformed_input_positions = (input_positions - target_position)[:, [0, 2]]
        if self.mode == 'train':
            rand_angle = np.random.uniform(-np.pi, np.pi)
            rot_matrix = np.array([[np.cos(rand_angle), -np.sin(rand_angle)], [np.sin(rand_angle), np.cos(rand_angle)]])
            transformed_input_positions = transformed_input_positions @ rot_matrix.T
        return transformed_input_positions
    
    def select_target_pose(self, future_poses):
        arrived = np.random.rand() < self.arrived_prob
        if arrived:
            target_idx = random.randint(self.wp_length, self.wp_length + self.arrived_threshold)
        else:
            target_idx = random.randint(self.wp_length + self.arrived_threshold, future_poses.shape[0] - 1)
        target_pose = future_poses[target_idx]
        return target_pose, arrived

    # def determine_arrived_label(self, current_pos, target_pos):
    #     distance_to_goal = np.linalg.norm(target_pos - current_pos, axis=0)
    #     arrived = distance_to_goal <= self.arrived_threshold
    #     return arrived

    def extract_waypoints(self, pose, pose_start):
        waypoint_start = pose_start + self.context_size
        waypoint_end = waypoint_start + self.wp_length
        waypoint_poses = pose[waypoint_start: waypoint_end]
        return waypoint_poses

    # def add_noise(self, input_poses):
    #     noise = np.random.normal(0, self.input_noise, input_poses[:, :3].shape)
    #     scale = np.linalg.norm(input_poses[-1, :3] - input_poses[-2, :3])
    #     input_poses[:, :3] += noise * scale
    #     return input_poses

    def process_frames(self, frames):
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0  # Corrected normalization
        # Desired resolution
        desired_height = 360
        desired_width = 640
        
        # Current resolution
        _, _, H, W = frames.shape
        
        # Calculate padding needed
        pad_height = desired_height - H
        pad_width = desired_width - W
        
        # Only pad if necessary
        if pad_height > 0 or pad_width > 0:
            # Calculate padding for each side (left, right, top, bottom)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            # Apply padding
            frames = TF.pad(
                frames, 
                (pad_left, pad_top, pad_right, pad_bottom),
            )
            
            # Optional: Verify the new shape
            assert frames.shape[2] == desired_height and frames.shape[3] == desired_width, \
                f"Padded frames have incorrect shape: {frames.shape}. Expected ({desired_height}, {desired_width})"
            
        if pad_height < 0  or pad_width < 0:
            frames = TF.center_crop(frames, (desired_height, desired_width))
        
        return frames

    def transform_waypoints(self, waypoint_poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        waypoint_matrices = self.poses_to_matrices(waypoint_poses)
        transformed_waypoint_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], waypoint_matrices)
        waypoints_positions = transformed_waypoint_matrices[:, :3, 3]
        return waypoints_positions

    def transform_target_pose(self, target_pose, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        target_pose_matrix = self.pose_to_matrix(target_pose)
        transformed_target_matrix = np.matmul(current_pose_inv, target_pose_matrix)
        target_position = transformed_target_matrix[:3, 3]
        return target_position

    def pose_to_matrix(self, pose):
        position = pose[:3]
        rotation = R.from_quat(pose[3:])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix

    def poses_to_matrices(self, poses):
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices
