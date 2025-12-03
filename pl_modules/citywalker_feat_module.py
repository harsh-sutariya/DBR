import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from model.citywalker_feat import CityWalkerFeat
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8')
import os

class CityWalkerFeatModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CityWalkerFeat(cfg)
        self.save_hyperparameters(cfg)
        self.do_normalize = cfg.training.normalize_step_length
        self.datatype = cfg.data.type
        
        # Coordinate representation
        self.output_coordinate_repr = cfg.model.output_coordinate_repr
        if self.output_coordinate_repr not in ["euclidean"]:
            raise ValueError(f"Unsupported coordinate representation: {self.output_coordinate_repr}")
        
        self.decoder = cfg.model.decoder.type
        if self.decoder not in ["attention"]:
            raise ValueError(f"Unsupported decoder: {self.decoder}")
        
        # Direction loss weight (you can adjust this value in your cfg)
        self.direction_loss_weight = cfg.training.direction_loss_weight
        self.feature_loss_weight = cfg.training.feature_loss_weight
        
        # DBR support
        self.use_dbr = getattr(cfg.model, 'use_dbr', False)
        
        # Visualization settings
        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

        if self.datatype == "teleop":
            self.test_catetories = ['crowd', 'person_close_by', 'turn', 'action_target_mismatch', 'crossing', 'other']
            self.num_categories = len(self.test_catetories)

    def forward(self, obs, cord, future_obs, depth_map=None, depth_mask=None):
        return self.model(obs, cord, future_obs, depth_map, depth_mask)
    
    def training_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        if "future_video_frames" in batch:
            future_obs = batch['future_video_frames']
        else:
            future_obs = None
        
        # Extract depth data - only used if DBR is enabled during training
        # Depth is also loaded for val/test even when use_dbr=False for evaluation metrics
        depth_map = batch.get('depth_map', None) if self.use_dbr else None
        depth_mask = batch.get('depth_mask', None) if self.use_dbr else None
        
        wp_pred, arrive_pred, feature_pred, feature_gt = self(obs, cord, future_obs, depth_map, depth_mask)
        losses = self.compute_loss(wp_pred, arrive_pred, feature_pred, feature_gt, batch)
        waypoints_loss = losses['waypoints_loss']
        arrived_loss = losses['arrived_loss']
        direction_loss = losses['direction_loss']
        feature_loss = losses['feature_loss']
        total_loss = waypoints_loss + arrived_loss + self.direction_loss_weight * direction_loss + feature_loss * self.feature_loss_weight
        
        # Add DBR loss if enabled
        if self.use_dbr and depth_map is not None:
            # Scale waypoints back to metric space for DBR
            step_scale = batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            wp_pred_metric = wp_pred * step_scale
            
            # Get depth resize ratios for intrinsics adjustment (if available)
            # Resize ratios are stored as Python floats, but may be batched
            resize_ratio_h = batch.get('depth_resize_ratio_h', None)
            resize_ratio_w = batch.get('depth_resize_ratio_w', None)
            # Extract scalar value if batched (take first element, assume all samples have same ratio)
            if resize_ratio_h is not None:
                if isinstance(resize_ratio_h, torch.Tensor):
                    resize_ratio_h = resize_ratio_h[0].item() if resize_ratio_h.numel() > 0 else None
                elif isinstance(resize_ratio_h, (list, tuple)):
                    resize_ratio_h = resize_ratio_h[0] if len(resize_ratio_h) > 0 else None
            if resize_ratio_w is not None:
                if isinstance(resize_ratio_w, torch.Tensor):
                    resize_ratio_w = resize_ratio_w[0].item() if resize_ratio_w.numel() > 0 else None
                elif isinstance(resize_ratio_w, (list, tuple)):
                    resize_ratio_w = resize_ratio_w[0] if len(resize_ratio_w) > 0 else None
            
            dbr_loss, clearance_vector = self.model.dbr_module(
                wp_pred_metric, depth_map, depth_mask,
                resize_ratio_h=resize_ratio_h, resize_ratio_w=resize_ratio_w
            )
            total_loss = total_loss + dbr_loss
            self.log('train/l_dbr', dbr_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            # Log average clearance for monitoring
            avg_clearance = clearance_vector.mean()
            self.log('train/avg_clearance', avg_clearance, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            
            # Compute and log DBR metrics (DVR and MDM) for monitoring
            # Get depth resize ratios for intrinsics adjustment (if available)
            resize_ratio_h_train = batch.get('depth_resize_ratio_h', None)
            resize_ratio_w_train = batch.get('depth_resize_ratio_w', None)
            if resize_ratio_h_train is not None:
                if isinstance(resize_ratio_h_train, torch.Tensor):
                    resize_ratio_h_train = resize_ratio_h_train[0].item() if resize_ratio_h_train.numel() > 0 else None
                elif isinstance(resize_ratio_h_train, (list, tuple)):
                    resize_ratio_h_train = resize_ratio_h_train[0] if len(resize_ratio_h_train) > 0 else None
            if resize_ratio_w_train is not None:
                if isinstance(resize_ratio_w_train, torch.Tensor):
                    resize_ratio_w_train = resize_ratio_w_train[0].item() if resize_ratio_w_train.numel() > 0 else None
                elif isinstance(resize_ratio_w_train, (list, tuple)):
                    resize_ratio_w_train = resize_ratio_w_train[0] if len(resize_ratio_w_train) > 0 else None
            
            dvr, mdm = self.compute_dbr_metrics(
                wp_pred_metric, depth_map, depth_mask,
                resize_ratio_h=resize_ratio_h_train, resize_ratio_w=resize_ratio_w_train
            )
            if dvr is not None and mdm is not None:
                self.log('train/dvr', dvr, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('train/mdm', mdm, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Common logs
        self.log('train/l_wp', waypoints_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_arvd', arrived_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_dir', direction_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_feat', feature_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        if "future_video_frames" in batch:
            future_obs = batch['future_video_frames']
        else:
            future_obs = None
        
        # Extract depth data if DBR is enabled OR if available for evaluation metrics
        depth_map = batch.get('depth_map', None)
        depth_mask = batch.get('depth_mask', None)
        
        wp_pred, arrive_pred, feature_pred, feature_gt = self(obs, cord, future_obs, depth_map, depth_mask)
        losses = self.compute_loss(wp_pred, arrive_pred, feature_pred, feature_gt, batch)
        l1_loss = losses['waypoints_loss']
        direction_loss = losses['direction_loss']
        feature_loss = losses['feature_loss']
        self.log('val/l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Compute accuracy for "arrived" prediction
        arrived_target = batch['arrived']
        arrived_logits = arrive_pred.flatten()
        arrived_probs = torch.sigmoid(arrived_logits)
        arrived_pred_binary = (arrived_probs >= 0.5).float()
        correct = (arrived_pred_binary == arrived_target).float()
        accuracy = correct.sum() / correct.numel()
        
        # Log the metrics
        self.log('val/arrived_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/direction_loss', direction_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/feature_loss', feature_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Compute and log DBR metrics if depth is available (even when use_dbr is False, for comparison)
        if depth_map is not None:
            # Scale waypoints to metric space for DBR evaluation
            step_scale = batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            wp_pred_metric = wp_pred * step_scale
            
            # Get depth resize ratios for intrinsics adjustment (if available)
            resize_ratio_h = batch.get('depth_resize_ratio_h', None)
            resize_ratio_w = batch.get('depth_resize_ratio_w', None)
            if resize_ratio_h is not None and isinstance(resize_ratio_h, torch.Tensor):
                resize_ratio_h = resize_ratio_h[0].item() if resize_ratio_h.numel() > 0 else None
            if resize_ratio_w is not None and isinstance(resize_ratio_w, torch.Tensor):
                resize_ratio_w = resize_ratio_w[0].item() if resize_ratio_w.numel() > 0 else None
            
            dvr, mdm = self.compute_dbr_metrics(wp_pred_metric, depth_map, depth_mask, 
                                                 resize_ratio_h=resize_ratio_h, resize_ratio_w=resize_ratio_w)
            if dvr is not None and mdm is not None:
                self.log('val/dvr', dvr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('val/mdm', mdm, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Handle visualization
        wp_pred_vis = wp_pred * batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
        self.process_visualization(
            mode='val',
            batch=batch,
            obs=obs,
            wp_pred=wp_pred_vis,
            arrive_pred=arrive_pred
        )
        
        return direction_loss

    def test_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        if "future_video_frames" in batch:
            future_obs = batch['future_video_frames']
        else:
            future_obs = None
        B, T, _, _, _ = obs.shape
        
        # Extract depth data if DBR is enabled OR if available for evaluation metrics
        depth_map = batch.get('depth_map', None)
        depth_mask = batch.get('depth_mask', None)
        
        # Initialize variables to avoid UnboundLocalError
        wp_pred = None
        arrive_pred = None
        
        if self.datatype == "citywalk" or self.datatype == "citywalk_feat":
            wp_pred, arrive_pred, _, _ = self(obs, cord, future_obs, depth_map, depth_mask)
            # Compute L1 loss for waypoints
            waypoints_target = batch['waypoints']
            l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='mean').item()
            
            # Compute accuracy for "arrived" prediction
            arrived_target = batch['arrived']
            arrived_logits = arrive_pred.flatten()
            arrived_probs = torch.sigmoid(arrived_logits)
            arrived_pred_binary = (arrived_probs >= 0.5).float()
            correct = (arrived_pred_binary == arrived_target).float()
            accuracy = correct.sum().item() / correct.numel()

            # wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
            # waypoints_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

            # Compute cosine similarity
            wp_pred_view = wp_pred.view(-1, 2)
            waypoints_target_view = waypoints_target.view(-1, 2)
            # dot_product = (wp_pred_view * waypoints_target_view).sum(dim=1)  # shape [batch_size]
            # norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
            # norm_target = waypoints_target_view.norm(dim=1)  # shape [batch_size]
            # cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero
            cos_sim = F.cosine_similarity(wp_pred_view, waypoints_target_view, dim=1)
            
            # Compute angle in degrees
            angle = torch.acos(cos_sim) * 180 / torch.pi  # shape [batch_size]
            angle = angle.view(B, T)
            
            # Take mean angle
            mean_angle = angle.mean(dim=0).cpu().numpy()
            
            # Compute DBR metrics if depth is available (even when use_dbr is False, for comparison)
            if depth_map is not None:
                # Scale waypoints to metric space for DBR evaluation
                step_scale = batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
                wp_pred_metric = wp_pred * step_scale
                
                # Get depth resize ratios for intrinsics adjustment (if available)
                resize_ratio_h_test = batch.get('depth_resize_ratio_h', None)
                resize_ratio_w_test = batch.get('depth_resize_ratio_w', None)
                if resize_ratio_h_test is not None:
                    if isinstance(resize_ratio_h_test, torch.Tensor):
                        resize_ratio_h_test = resize_ratio_h_test[0].item() if resize_ratio_h_test.numel() > 0 else None
                    elif isinstance(resize_ratio_h_test, (list, tuple)):
                        resize_ratio_h_test = resize_ratio_h_test[0] if len(resize_ratio_h_test) > 0 else None
                if resize_ratio_w_test is not None:
                    if isinstance(resize_ratio_w_test, torch.Tensor):
                        resize_ratio_w_test = resize_ratio_w_test[0].item() if resize_ratio_w_test.numel() > 0 else None
                    elif isinstance(resize_ratio_w_test, (list, tuple)):
                        resize_ratio_w_test = resize_ratio_w_test[0] if len(resize_ratio_w_test) > 0 else None
                
                dvr, mdm = self.compute_dbr_metrics(
                    wp_pred_metric, depth_map, depth_mask,
                    resize_ratio_h=resize_ratio_h_test, resize_ratio_w=resize_ratio_w_test
                )
                if dvr is not None and mdm is not None:
                    self.test_metrics['dvr'].append(dvr)
                    self.test_metrics['mdm'].append(mdm)
            
            # Store the metrics
            if self.output_coordinate_repr == "euclidean":
                self.test_metrics['l1_loss'].append(l1_loss)
            self.test_metrics['arrived_accuracy'].append(accuracy)
            self.test_metrics['mean_angle'].append(mean_angle)
        elif self.datatype == "teleop":
            category = batch['categories']
            wp_pred, arrive_pred, _, _ = self(obs, cord, future_obs, depth_map, depth_mask)
            wp_pred *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            
            # Compute L1 loss for waypoints
            waypoints_target = batch['waypoints']
            waypoints_target *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            # l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='none')
            l1_loss = F.mse_loss(wp_pred, waypoints_target, reduction='none') ** 0.5
            
            # Compute accuracy for "arrived" prediction
            arrived_target = batch['arrived']
            arrived_probs = torch.sigmoid(arrive_pred)
            arrived_pred_binary = (arrived_probs >= 0.5).float().squeeze(-1)
            correct = (arrived_pred_binary == arrived_target).float()

            # Compute cosine similarity
            wp_pred_view = wp_pred.view(-1, 2)
            waypoints_target_view = waypoints_target.view(-1, 2)
            # dot_product = (wp_pred_view * waypoints_target_view).sum(dim=1)  # shape [batch_size]
            # norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
            # norm_target = waypoints_target_view.norm(dim=1)  # shape [batch_size]
            # cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero
            cos_sim = F.cosine_similarity(wp_pred_view, waypoints_target_view, dim=1)
            # Compute angle in degrees
            angle = torch.acos(cos_sim) * 180 / torch.pi  # shape [batch_size]
            angle = angle.view(B, T)

            gt_wp_last_norm = waypoints_target[:, -1, :].norm(dim=1)

            # Compute DBR metrics if depth is available (for teleop dataset, even when use_dbr is False)
            batch_dvr = None
            batch_mdm = None
            if depth_map is not None:
                # Compute DBR metrics per batch item
                batch_dvr_list = []
                batch_mdm_list = []
                # Get depth resize ratios for intrinsics adjustment (if available)
                resize_ratio_h_teleop = batch.get('depth_resize_ratio_h', None)
                resize_ratio_w_teleop = batch.get('depth_resize_ratio_w', None)
                if resize_ratio_h_teleop is not None:
                    if isinstance(resize_ratio_h_teleop, torch.Tensor):
                        resize_ratio_h_teleop = resize_ratio_h_teleop[0].item() if resize_ratio_h_teleop.numel() > 0 else None
                    elif isinstance(resize_ratio_h_teleop, (list, tuple)):
                        resize_ratio_h_teleop = resize_ratio_h_teleop[0] if len(resize_ratio_h_teleop) > 0 else None
                if resize_ratio_w_teleop is not None:
                    if isinstance(resize_ratio_w_teleop, torch.Tensor):
                        resize_ratio_w_teleop = resize_ratio_w_teleop[0].item() if resize_ratio_w_teleop.numel() > 0 else None
                    elif isinstance(resize_ratio_w_teleop, (list, tuple)):
                        resize_ratio_w_teleop = resize_ratio_w_teleop[0] if len(resize_ratio_w_teleop) > 0 else None
                
                for batch_idx in range(B):
                    wp_pred_batch = wp_pred[batch_idx:batch_idx+1]  # (1, T, 2)
                    depth_map_batch = depth_map[batch_idx:batch_idx+1]  # (1, H, W)
                    depth_mask_batch = depth_mask[batch_idx:batch_idx+1] if depth_mask is not None else None
                    dvr, mdm = self.compute_dbr_metrics(
                        wp_pred_batch, depth_map_batch, depth_mask_batch,
                        resize_ratio_h=resize_ratio_h_teleop, resize_ratio_w=resize_ratio_w_teleop
                    )
                    if dvr is not None and mdm is not None:
                        batch_dvr_list.append(dvr)
                        batch_mdm_list.append(mdm)
                if batch_dvr_list:
                    batch_dvr = np.array(batch_dvr_list)
                    batch_mdm = np.array(batch_mdm_list)

            for batch_idx in range(B):
                for category_idx in range(self.num_categories):
                    if category[batch_idx, category_idx] == 1:
                        category_name = self.test_catetories[category_idx]
                        self.test_metrics[category_name]['l1_loss'].append(l1_loss[batch_idx].max().item())
                        self.test_metrics[category_name]['arrived_accuracy'].append(correct[batch_idx].item())
                        if gt_wp_last_norm[batch_idx] > 1:
                            self.test_metrics[category_name]['mean_angle'].append(angle[batch_idx].max().item())
                            self.test_metrics[category_name]['angle_step1'].append(angle[batch_idx, 0].item())
                            self.test_metrics[category_name]['angle_step2'].append(angle[batch_idx, 1].item())
                            self.test_metrics[category_name]['angle_step3'].append(angle[batch_idx, 2].item())
                            self.test_metrics[category_name]['angle_step4'].append(angle[batch_idx, 3].item())
                            self.test_metrics[category_name]['angle_step5'].append(angle[batch_idx, 4].item())
                        # Add DBR metrics if available
                        if batch_dvr is not None and batch_mdm is not None:
                            self.test_metrics[category_name]['dvr'].append(batch_dvr[batch_idx])
                            self.test_metrics[category_name]['mdm'].append(batch_mdm[batch_idx])
                    else:
                        continue
                self.test_metrics['overall']['l1_loss'].append(l1_loss[batch_idx].max().item())
                self.test_metrics['overall']['arrived_accuracy'].append(correct[batch_idx].item())
                if gt_wp_last_norm[batch_idx] > 1:
                    self.test_metrics['overall']['mean_angle'].append(angle[batch_idx].max().item())
                    self.test_metrics['overall']['angle_step1'].append(angle[batch_idx, 0].item())
                    self.test_metrics['overall']['angle_step2'].append(angle[batch_idx, 1].item())
                    self.test_metrics['overall']['angle_step3'].append(angle[batch_idx, 2].item())
                    self.test_metrics['overall']['angle_step4'].append(angle[batch_idx, 3].item())
                    self.test_metrics['overall']['angle_step5'].append(angle[batch_idx, 4].item())
                # Add DBR metrics if available
                if batch_dvr is not None and batch_mdm is not None:
                    self.test_metrics['overall']['dvr'].append(batch_dvr[batch_idx])
                    self.test_metrics['overall']['mdm'].append(batch_mdm[batch_idx])

        
        # Handle visualization
        if wp_pred is not None and arrive_pred is not None:
            if self.datatype == "citywalk" or self.datatype == "citywalk_feat":
                wp_pred *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            if self.output_coordinate_repr == "euclidean":
                self.process_visualization(
                    mode='test',
                    batch=batch,
                    obs=obs,
                    wp_pred=wp_pred,
                    arrive_pred=arrive_pred
                )
            elif self.output_coordinate_repr == "polar":
                self.process_visualization(
                    mode='test',
                    batch=batch,
                    obs=obs,
                    wp_pred=wp_pred,
                    arrive_pred=arrive_pred
                )

    def on_test_epoch_end(self):
        if self.datatype == "citywalk" or self.datatype == "citywalk_feat":
            for metric in self.test_metrics:
                metric_array = np.array(self.test_metrics[metric])
                save_path = os.path.join(self.result_dir, f'test_{metric}.npy')
                np.save(save_path, metric_array)
                if not metric == "mean_angle":
                    mean_val = metric_array.mean()
                    print(f"Test mean {metric} {mean_val:.4f} saved to {save_path}")
                    # Log to WandB
                    self.log(f'test/{metric}', mean_val, on_step=False, on_epoch=True, sync_dist=True)
                else:
                    mean_angle = metric_array.mean(axis=0)
                    for i in range(len(mean_angle)):
                        print(f"Test mean angle at step {i} {mean_angle[i]:.4f}")
                        # Log each step's angle to WandB
                        self.log(f'test/angle_step{i}', mean_angle[i], on_step=False, on_epoch=True, sync_dist=True)
                    # Also log mean of all steps
                    self.log('test/mean_angle_all', mean_angle.mean(), on_step=False, on_epoch=True, sync_dist=True)
            # Print DBR metrics summary if available (already logged above in loop)
            if 'dvr' in self.test_metrics and 'mdm' in self.test_metrics:
                dvr_array = np.array(self.test_metrics['dvr'])
                mdm_array = np.array(self.test_metrics['mdm'])
                if len(dvr_array) > 0:
                    dvr_mean = dvr_array.mean()
                    mdm_mean = mdm_array.mean()
                    print(f"Test mean DVR (Depth Violation Rate): {dvr_mean:.4f}%")
                    print(f"Test mean MDM (Min-Depth Margin): {mdm_mean:.4f}m")
        elif self.datatype == "teleop":
            import pandas as pd
            for category in self.test_catetories:
                # Add a new 'count' metric for each category by counting 'l1_loss' entries
                self.test_metrics[category]['count'] = len(self.test_metrics[category]['l1_loss'])
            self.test_metrics['overall']['count'] = sum(self.test_metrics[category]['count'] for category in self.test_catetories)
            self.test_metrics['mean']['count'] = 0

            for category in self.test_catetories:
                for metric in self.test_metrics[category]:
                    if metric != 'count':
                        # print(f"{category} {metric}: {self.test_metrics[category][metric]}")
                        self.test_metrics[category][metric] = np.nanmean(np.array(self.test_metrics[category][metric]))
            for metric in self.test_metrics['overall']:
                if metric != 'count':
                    self.test_metrics['overall'][metric] = np.nanmean(np.array(self.test_metrics['overall'][metric]))
            metrics = ['l1_loss', 'arrived_accuracy', 'angle_step1', 'angle_step2', 'angle_step3', 'angle_step4', 'angle_step5', 'mean_angle']
            # Always include DBR metrics if available (even when use_dbr is False, for comparison)
            if 'dvr' in self.test_metrics.get('overall', {}):
                metrics.extend(['dvr', 'mdm'])
            for metric in metrics:
                category_val = []
                for category in self.test_catetories:
                    if metric in self.test_metrics[category]:
                        category_val.append(self.test_metrics[category][metric])
                if category_val:
                    self.test_metrics['mean'][metric] = np.array(category_val).mean()
                    overall_val = self.test_metrics['overall'][metric]
                    mean_val = self.test_metrics['mean'][metric]
                    print(f"{metric}: Sample mean {overall_val:.4f}, Category mean {mean_val:.4f}")
                    # Log to WandB
                    self.log(f'test/{metric}_overall', overall_val, on_step=False, on_epoch=True, sync_dist=True)
                    self.log(f'test/{metric}_mean', mean_val, on_step=False, on_epoch=True, sync_dist=True)
                # Also log per-category metrics
                for category in self.test_catetories:
                    if metric in self.test_metrics[category]:
                        category_val = self.test_metrics[category][metric]
                        self.log(f'test/{category}/{metric}', category_val, on_step=False, on_epoch=True, sync_dist=True)

            df = pd.DataFrame(self.test_metrics)
            df = df.reset_index().rename(columns={'index': 'Metrics'})
            save_path = os.path.join(self.result_dir, 'test_metrics.csv')
            df.to_csv(save_path, index=False)


    def on_validation_epoch_start(self):
        self.vis_count = 0

    def on_test_epoch_start(self):
        self.vis_count = 0
        if self.datatype == "citywalk" or self.datatype == "citywalk_feat":
            if self.output_coordinate_repr == "euclidean":
                self.test_metrics = {'l1_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
                # Always add DBR metrics (even when use_dbr is False, for comparison)
                self.test_metrics['dvr'] = []  # Depth Violation Rate
                self.test_metrics['mdm'] = []  # Min-Depth Margin
            elif self.output_coordinate_repr == "polar":
                self.test_metrics = {'distance_loss': [], 'angle_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
                # Always add DBR metrics (even when use_dbr is False, for comparison)
                self.test_metrics['dvr'] = []
                self.test_metrics['mdm'] = []
        elif self.datatype == "teleop":
            self.test_metrics = {}
            categories = self.test_catetories[:]
            categories.extend(['mean', 'overall'])
            for category in categories:
                if self.output_coordinate_repr == "euclidean":
                    self.test_metrics[category] = {
                        'l1_loss': [], 
                        'arrived_accuracy': [], 
                        'angle_step1': [],
                        'angle_step2': [],
                        'angle_step3': [],
                        'angle_step4': [],
                        'angle_step5': [],
                        'mean_angle': []
                    }
                    # Always add DBR metrics (even when use_dbr is False, for comparison)
                    self.test_metrics[category]['dvr'] = []
                    self.test_metrics[category]['mdm'] = []
                elif self.output_coordinate_repr == "polar":
                    raise ValueError("Polar representation is not supported for teleop dataset.")

    def compute_dbr_metrics(self, wp_pred_metric, depth_map, depth_mask, resize_ratio_h=None, resize_ratio_w=None):
        """
        Compute DBR safety metrics: DVR (Depth Violation Rate) and MDM (Min-Depth Margin).
        
        This works even when use_dbr is False - it creates a temporary DBR module for evaluation.
        This allows comparing baseline models (no DBR training) with DBR-trained models.
        
        Args:
            wp_pred_metric: (B, T, 2) predicted waypoints in metric space
            depth_map: (B, H, W) depth values in meters
            depth_mask: (B, H, W) optional mask for valid depth pixels
            resize_ratio_h: Optional height resize ratio for intrinsics adjustment
            resize_ratio_w: Optional width resize ratio for intrinsics adjustment
            
        Returns:
            dvr: Depth Violation Rate (% of steps where d_min < margin)
            mdm: Min-Depth Margin (mean d_min across all waypoints)
        """
        if depth_map is None:
            return None, None
        
        # Get DBR module if available, otherwise create a temporary one for evaluation
        if self.use_dbr and hasattr(self.model, 'dbr_module'):
            dbr_module = self.model.dbr_module
            polar_reducer = dbr_module.polar_reducer
            barrier_loss = dbr_module.barrier_loss
            margin = barrier_loss.margin
        else:
            # Create a temporary DBR module for evaluation only
            # Use default DBR config values
            from model.dbr import DBRModule
            import types
            
            # Create a minimal config object with DBR defaults
            class DBRConfig:
                def __init__(self):
                    self.num_bins = 32
                    self.temperature = 20.0
                    self.crop_bottom_ratio = 0.6
                    self.fov_horizontal = 90.0
                    self.margin = 0.5  # Default safety margin
                    self.weight = 1.0
                    self.fx = 320.0
                    self.fy = 320.0
                    self.cx = 320.0
                    self.cy = 180.0
            
            class ModelConfig:
                def __init__(self):
                    self.dbr = DBRConfig()
            
            class TempConfig:
                def __init__(self):
                    self.model = ModelConfig()
            
            temp_cfg = TempConfig()
            temp_dbr_module = DBRModule(temp_cfg)
            temp_dbr_module.eval()  # Set to eval mode
            if depth_map.is_cuda:
                temp_dbr_module = temp_dbr_module.cuda()
            
            polar_reducer = temp_dbr_module.polar_reducer
            barrier_loss = temp_dbr_module.barrier_loss
            margin = barrier_loss.margin
        
        # Convert depth to polar clearance vector with adjusted intrinsics
        clearance_vector, bin_centers = polar_reducer(
            depth_map, depth_mask, 
            resize_ratio_h=resize_ratio_h, resize_ratio_w=resize_ratio_w
        )
        
        # Compute yaw angles for each waypoint
        B, T, _ = wp_pred_metric.shape
        yaw_angles = torch.atan2(wp_pred_metric[:, :, 1], wp_pred_metric[:, :, 0])  # (B, T)
        
        # Interpolate clearance at each waypoint direction
        d_min = barrier_loss.bilinear_interpolate(clearance_vector, bin_centers, yaw_angles)  # (B, T)
        
        # Compute DVR: % of waypoints where d_min < margin
        violations = (d_min < margin).float()  # (B, T)
        dvr = violations.mean().item() * 100.0  # Convert to percentage
        
        # Compute MDM: mean d_min across all waypoints
        mdm = d_min.mean().item()
        
        return dvr, mdm

    def compute_loss(self, wp_pred, arrive_pred, feature_pred, feature_gt, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        if feature_pred is not None and feature_gt is not None:
            feature_loss = F.mse_loss(feature_pred, feature_gt)
        else:
            feature_loss = 0.0
        wp_loss = F.l1_loss(wp_pred, waypoints_target)
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)

        # Compute direction loss
        wp_pred_view = wp_pred.view(-1, 2)
        wp_target_view = waypoints_target.view(-1, 2)

        # Compute cosine similarity
        dot_product = (wp_pred_view * wp_target_view).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_view.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = 1 - cos_sim.mean()

        return {'waypoints_loss': wp_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss, 'feature_loss': feature_loss}
    
    def compute_loss_polar(self, wp_pred_euclidean, distance_pred, angle_pred, arrive_pred, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        
        # Compute distance and angle targets
        distance_target, angle_target = self.waypoints_to_polar(waypoints_target)
        
        # Compute L1 loss for distance and angle
        distance_loss = F.l1_loss(distance_pred, distance_target)
        angle_loss = F.l1_loss(angle_pred, angle_target)
        
        # Compute arrived loss
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)
        
        # Compute direction loss using Euclidean waypoints
        wp_pred_last = wp_pred_euclidean[:, -1, :]  # shape [batch_size, 2]
        wp_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        dot_product = (wp_pred_last * wp_target_last).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_last.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_last.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = (1 - cos_sim.mean()) ** 2

        return {'distance_loss': distance_loss, 'angle_loss': angle_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss}

    def compute_loss_diff_policy(self, wp_pred, noise_pred, arrived_pred, noise, batch):
        # Compute loss for noise prediction
        waypoints_target = batch['waypoints']
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Compute loss for arrived prediction
        arrived_target = batch['arrived']
        arrived_loss = F.binary_cross_entropy_with_logits(arrived_pred.flatten(), arrived_target)

        # Compute direction loss
        wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
        wp_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        dot_product = (wp_pred_last * wp_target_last).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_last.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_last.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = (1 - cos_sim.mean()) ** 2
        
        return {'noise_loss': noise_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss}
    
    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        scheduler_cfg = self.cfg.scheduler
        if scheduler_cfg.name.lower() == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'none':
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")

    def process_visualization(self, mode, batch, obs, wp_pred, arrive_pred):
        """
        Handles visualization for both validation and testing.

        Args:
            mode (str): 'val' or 'test'
            batch (dict): Batch data
            obs (torch.Tensor): Observation frames
            wp_pred (torch.Tensor): Predicted waypoints
            arrive_pred (torch.Tensor): Predicted arrival logits
        """
        if mode == 'val':
            num_visualize = self.val_num_visualize
            vis_dir = os.path.join(self.result_dir, 'val_vis', f'epoch_{self.current_epoch}')
        elif mode == 'test':
            num_visualize = self.test_num_visualize
            vis_dir = os.path.join(self.result_dir, 'test_vis')
        else:
            raise ValueError("Mode should be either 'val' or 'test'.")

        os.makedirs(vis_dir, exist_ok=True)

        batch_size = obs.size(0)
        for idx in range(batch_size):
            if self.vis_count >= num_visualize:
                break

            # Extract necessary data
            arrived_target = batch['arrived'][idx].item()
            arrived_logits = arrive_pred[idx].flatten()
            arrived_probs = torch.sigmoid(arrived_logits).item()

            original_input_positions = batch['original_input_positions'][idx].cpu().numpy()
            noisy_input_positions = batch['noisy_input_positions'][idx].cpu().numpy()
            gt_waypoints = batch['gt_waypoints'][idx].cpu().numpy()
            pred_waypoints = wp_pred[idx].detach().cpu().numpy()
            target_transformed = batch['target_transformed'][idx].cpu().numpy()

            # if self.do_normalize:
            #     step_length = np.linalg.norm(gt_waypoints, axis=1).mean()
            #     original_input_positions = original_input_positions / step_length
            #     noisy_input_positions = noisy_input_positions / step_length
            #     gt_waypoints = gt_waypoints / step_length
            #     pred_waypoints = pred_waypoints / step_length
            #     target_transformed = target_transformed / step_length

            # Get the last frame from the sequence
            frame = obs[idx, -1].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8 for visualization

            # Visualization title
            arrive_title = f"Arrived GT: {'True' if arrived_target else 'False'}, Pred: {arrived_probs:.2f}"

                       # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plt.subplots_adjust(wspace=0.3)

            # Left axis: plot the current observation (frame) with arrived info in title
            ax1.imshow(frame)
            ax1.axis('off')
            ax1.set_title(arrive_title, fontsize=20)

            # Right axis: plot the coordinates
            ax2.axis('equal')
            ax2.plot(original_input_positions[:, 0], original_input_positions[:, 1],
                     'o-', label='Original Input Positions', color='#5771DB')
            ax2.plot(noisy_input_positions[:, 0], noisy_input_positions[:, 1],
                     'o-', label='Noisy Input Positions', color='#DBC257')
            ax2.plot(gt_waypoints[:, 0], gt_waypoints[:, 1],
                     'X-', label='GT Waypoints', color='#92DB58')
            ax2.plot(pred_waypoints[:, 0], pred_waypoints[:, 1],
                     's-', label='Predicted Waypoints', color='#DB6057')
            ax2.plot(target_transformed[0], target_transformed[1],
                     marker='*', markersize=15, label='Target Coordinate', color='#A157DB')
            # ax2.legend(fontsize=20)
            ax2.set_title('Coordinates', fontsize=20)
            ax2.set_xlabel('X (m)', fontsize=20)
            ax2.set_ylabel('Y (m)', fontsize=20)
            ax2.tick_params(axis='both', labelsize=18)
            ax2.grid(True)

            # Save the plot
            output_path = os.path.join(vis_dir, f'sample_{self.vis_count}.png')
            plt.savefig(output_path)
            plt.close(fig)

            self.vis_count += 1

    def waypoints_to_polar(self, waypoints):
        # Compute relative differences
        deltas = torch.diff(waypoints, dim=1, prepend=torch.zeros_like(waypoints[:, :1, :]))
        distance = torch.norm(deltas, dim=2)
        angle = torch.atan2(deltas[:, :, 1], deltas[:, :, 0])
        return distance, angle
