# main.py

import pytorch_lightning as pl
import argparse
import yaml
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.teleop_datamodule import TeleopDataModule
from pl_modules.citywalker_module import CityWalkerModule
from pl_modules.citywalker_feat_module import CityWalkerFeatModule
from pl_modules.citywalk_feat_datamodule import CityWalkFeatDataModule
from pytorch_lightning.strategies import DDPStrategy
import torch
import glob
torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)


# Remove the WandbLogger import from the top
# from pytorch_lightning.loggers import WandbLogger

class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)

def parse_args():
    parser = argparse.ArgumentParser(description='Train UrbanNav model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. If not provided, the latest checkpoint will be used.')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DictNamespace(**cfg_dict)
    
    # Apply quick_debug settings if enabled
    quick_debug = getattr(cfg.training, 'quick_debug', False)
    if quick_debug:
        print("="*50)
        print("QUICK_DEBUG MODE ENABLED - Using reduced settings for faster debugging")
        print("="*50)
        
        # Reduce training data
        cfg.data.num_train = min(cfg.data.num_train, 100)  # Max 100 samples
        cfg.data.num_val = min(cfg.data.num_val, 20)  # Max 20 samples
        # Set num_test to 10 if it's 0 or very large (for quick debug)
        current_num_test = getattr(cfg.data, 'num_test', 0)
        if current_num_test == 0 or current_num_test > 10:
            cfg.data.num_test = 10  # Use 10 samples for quick debug
        else:
            cfg.data.num_test = min(current_num_test, 10)  # Cap at 10 if already set
        
        # Reduce batch size
        cfg.training.batch_size = 1
        
        # Reduce number of workers
        cfg.data.num_workers = 1
        
        # Reduce max epochs
        cfg.training.max_epochs = 1
        
        # Reduce visualization
        cfg.validation.num_visualize = 5
        cfg.testing.num_visualize = 5
        
        # Disable WandB for quick debug (optional)
        # cfg.logging.enable_wandb = False
        
        # Increase progress bar refresh rate
        cfg.logging.pbar_rate = 1
        
        print(f"Quick debug settings:")
        print(f"  - num_train: {cfg.data.num_train}")
        print(f"  - num_val: {cfg.data.num_val}")
        print(f"  - num_test: {cfg.data.num_test}")
        print(f"  - batch_size: {cfg.training.batch_size}")
        print(f"  - num_workers: {cfg.data.num_workers}")
        print(f"  - max_epochs: {cfg.training.max_epochs}")
        print("="*50)
    
    return cfg

def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint in the given directory based on modification time.
    
    Args:
        checkpoint_dir (str): Path to the directory containing checkpoints.
    
    Returns:
        str: Path to the latest checkpoint file.
    
    Raises:
        FileNotFoundError: If no checkpoint files are found in the directory.
    """
    print(checkpoint_dir)
    checkpoint_pattern = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
    
    # Sort checkpoints by modification time (latest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    return latest_checkpoint

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Create result directory
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.project.result_dir = result_dir  # Update result_dir in cfg

    # Save config file in result directory
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg.__dict__, f)

    # Initialize the DataModule
    if cfg.data.type == 'citywalk':
        datamodule = CityWalkDataModule(cfg)
    elif cfg.data.type == 'teleop':
        datamodule = TeleopDataModule(cfg)
    elif cfg.data.type == 'citywalk_feat':
        datamodule = CityWalkFeatDataModule(cfg)
    else:
        raise ValueError(f"Invalid dataset type: {cfg.data.type}")

    # Initialize the model
    if cfg.model.type == 'citywalker':
        model = CityWalkerModule(cfg)
    elif cfg.model.type == 'citywalker_feat':
        model = CityWalkerFeatModule(cfg)
    else:
        raise ValueError(f"Invalid model: {cfg.model.type}")
    print(pl.utilities.model_summary.ModelSummary(model, max_depth=2))
        
    # Get quick_debug flag
    quick_debug = getattr(cfg.training, 'quick_debug', False)
    # Force boolean conversion in case YAML loaded it as string
    if isinstance(quick_debug, str):
        quick_debug = quick_debug.lower() in ('true', '1', 'yes')
    quick_debug = bool(quick_debug)
    
    # Initialize logger - disable WandB in quick_debug mode
    logger = False if quick_debug else None  # False disables all logging in quick_debug
    
    if not quick_debug:
        # Check if logging with Wandb is enabled in config (only if not quick_debug)
        # Support both top-level 'wandb' and 'logging.enable_wandb' for backward compatibility
        use_wandb = getattr(cfg, 'wandb', None)
        if use_wandb is None:
            use_wandb = getattr(cfg.logging, 'enable_wandb', False)

        if use_wandb:
            try:
                from pytorch_lightning.loggers import WandbLogger  # Import here to handle ImportError
                import wandb
                import time
                
                # Generate unique run ID to prevent any accidental resuming
                # Use job ID from environment if available, otherwise timestamp
                job_id = os.environ.get('SLURM_JOB_ID', None)
                if job_id:
                    run_id = f"{cfg.project.run_name}_{job_id}"
                    resume_mode = "allow"  # Use "allow" to handle existing runs gracefully
                else:
                    run_id = f"{cfg.project.run_name}_{int(time.time())}"
                    resume_mode = "never"  # Never resume for production runs
                
                wandb_logger = WandbLogger(
                    project=cfg.project.name,
                    name=cfg.project.run_name,  # Display name in UI
                    id=run_id,  # Unique ID to prevent resuming
                    save_dir=result_dir,
                    resume=resume_mode
                )
                logger = wandb_logger
                print(f"WandbLogger initialized with unique ID: {run_id}")
            except ImportError:
                print("Wandb is not installed. Skipping Wandb logging.")

    # Set up checkpoint callback - disable in quick_debug mode
    if quick_debug:
        checkpoint_callback = None  # No checkpointing in quick_debug mode
        print("Quick debug: Checkpoints disabled")
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(result_dir, 'checkpoints'),
            save_last=True,
            save_top_k=1,
            monitor='val/direction_loss',
        )

    num_gpu = torch.cuda.device_count()
    
    # Set up Trainer
    # Note: quick_debug already retrieved above for WandB logger setup
    
    # Limit training batches for quick debug
    limit_train_batches = None
    limit_val_batches = None
    check_val_every_n_epoch = None
    if quick_debug:
        limit_train_batches = 5  # Only train for 5 batches
        limit_val_batches = 2  # Only validate for 2 batches
        check_val_every_n_epoch = 1  # Validate every epoch (ensures validation runs)
        print(f"Quick debug: Limiting training to {limit_train_batches} batches, validation to {limit_val_batches} batches")
    
    if num_gpu > 1:
        trainer = pl.Trainer(
            default_root_dir=result_dir,
            max_epochs=cfg.training.max_epochs,
            logger=logger,  # Pass the logger (WandbLogger or None)
            devices=num_gpu,
            precision='16-mixed' if cfg.training.amp else 32,
            accelerator='gpu',
            callbacks=(
                [checkpoint_callback] if checkpoint_callback is not None else []
            ) + [pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate)],
            log_every_n_steps=1,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            enable_checkpointing=not quick_debug,  # Disable checkpointing in quick_debug
            strategy=DDPStrategy(find_unused_parameters=True)
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=result_dir,
            max_epochs=cfg.training.max_epochs,
            logger=logger,  # Pass the logger (WandbLogger or None)
            devices=num_gpu,
            precision='16-mixed' if cfg.training.amp else 32,
            accelerator='gpu',
            callbacks=(
                [checkpoint_callback] if checkpoint_callback is not None else []
            ) + [pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate)],
            log_every_n_steps=1,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            enable_checkpointing=not quick_debug,  # Disable checkpointing in quick_debug
        )

    if cfg.training.resume:
        # Determine the checkpoint path
        try:
            if args.checkpoint:
                checkpoint_path = args.checkpoint
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            else:
                # Automatically find the latest checkpoint
                checkpoint_dir = os.path.join(cfg.project.result_dir, 'checkpoints')
                if not os.path.isdir(checkpoint_dir):
                    raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
                checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError()
                else:
                    print(f"No checkpoint specified. Using the latest checkpoint: {checkpoint_path}")
            print(f"Training resume from checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            print("No checkpoint found. Training from scratch.")
            checkpoint_path = None
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        # Start training
        trainer.fit(model, datamodule=datamodule)
    
    # Run testing after training completes (if enabled in config)
    test_after_training = getattr(cfg.training, 'test_after_training', False)
    quick_debug = getattr(cfg.training, 'quick_debug', False)
    
    # In quick_debug mode, always run validation/test at the end
    if quick_debug:
        print("\n" + "="*50)
        print("Quick debug mode: Running validation/test...")
        print("="*50)
        test_after_training = True
    
    if test_after_training:
        print("\n" + "="*50)
        print("Training completed. Starting testing...")
        print("="*50)
        
        # In quick_debug mode, use the in-memory model (no checkpoint loading)
        if quick_debug:
            print("Quick debug: Using in-memory model for testing (no checkpoint loading)")
            test_model = model  # Use the model that was just trained
        else:
            # Find the best checkpoint for testing
            checkpoint_dir = os.path.join(result_dir, 'checkpoints')
            try:
                best_checkpoint = find_latest_checkpoint(checkpoint_dir)
                print(f"Using checkpoint for testing: {best_checkpoint}")
                
                # Load the best model for testing
                if cfg.model.type == 'citywalker':
                    test_model = CityWalkerModule.load_from_checkpoint(best_checkpoint, cfg=cfg)
                elif cfg.model.type == 'citywalker_feat':
                    test_model = CityWalkerFeatModule.load_from_checkpoint(best_checkpoint, cfg=cfg)
                else:
                    raise ValueError(f"Invalid model type: {cfg.model.type}")
            except FileNotFoundError as e:
                print(f"Could not find checkpoint for testing: {e}")
                print("Using in-memory model instead.")
                test_model = model
        
        # Set up test directory
        test_dir = os.path.join(result_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        test_model.result_dir = test_dir
        
        # Create a new trainer specifically for testing (consistent with training trainer)
        limit_test_batches = None
        if quick_debug:
            limit_test_batches = 2  # Only test for 2 batches in quick debug
        
        if num_gpu > 1:
            test_trainer = pl.Trainer(
                default_root_dir=test_dir,
                devices=num_gpu,
                precision='16-mixed' if cfg.training.amp else 32,
                accelerator='gpu',
                logger=False,
                limit_test_batches=limit_test_batches,
                strategy=DDPStrategy(find_unused_parameters=True)
            )
        else:
            test_trainer = pl.Trainer(
                default_root_dir=test_dir,
                devices=num_gpu,
                precision='16-mixed' if cfg.training.amp else 32,
                accelerator='gpu',
                logger=False,
                limit_test_batches=limit_test_batches
            )
            
        # Run testing
        test_trainer.test(test_model, datamodule=datamodule, verbose=True)
        print(f"Testing completed. Results saved to: {test_dir}")
    else:
        print("\nTraining completed. Set 'test_after_training: true' in config to run testing automatically.")

if __name__ == '__main__':
    main()
