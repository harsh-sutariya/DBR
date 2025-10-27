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
        
    # Initialize logger
    logger = None  # Default to no logger

    # Check if logging with Wandb is enabled in config
    use_wandb = cfg.logging.enable_wandb

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger  # Import here to handle ImportError
            import wandb
            
            # Generate unique run ID to prevent any accidental resuming
            # Use job ID from environment if available, otherwise timestamp
            job_id = os.environ.get('SLURM_JOB_ID', None)
            if job_id:
                run_id = f"{cfg.project.run_name}_{job_id}"
            else:
                import time
                run_id = f"{cfg.project.run_name}_{int(time.time())}"
            
            wandb_logger = WandbLogger(
                project=cfg.project.name,
                name=cfg.project.run_name,  # Display name in UI
                id=run_id,  # Unique ID to prevent resuming
                save_dir=result_dir,
                resume="never"  # Force new run instead of resuming existing one
            )
            logger = wandb_logger
            print(f"WandbLogger initialized with unique ID: {run_id}")
        except ImportError:
            print("Wandb is not installed. Skipping Wandb logging.")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(result_dir, 'checkpoints'),
        save_last=True,
        save_top_k=1,
        monitor='val/direction_loss',
    )

    num_gpu = torch.cuda.device_count()
    # num_gpu = 1
    
    # Set up Trainer
    if num_gpu > 1:
        trainer = pl.Trainer(
            default_root_dir=result_dir,
            max_epochs=cfg.training.max_epochs,
            logger=logger,  # Pass the logger (WandbLogger or None)
            devices=num_gpu,
            precision='16-mixed' if cfg.training.amp else 32,
            accelerator='gpu',
            callbacks=[
                checkpoint_callback,
                pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate),
            ],
            log_every_n_steps=1,
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
            callbacks=[
                checkpoint_callback,
                pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate),
            ],
            log_every_n_steps=1,
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
    if test_after_training:
        print("\n" + "="*50)
        print("Training completed. Starting testing...")
        print("="*50)
        
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
            
            # Set up test directory
            test_dir = os.path.join(result_dir, 'test')
            os.makedirs(test_dir, exist_ok=True)
            test_model.result_dir = test_dir
            
            # Create a new trainer specifically for testing (consistent with training trainer)
            if num_gpu > 1:
                test_trainer = pl.Trainer(
                    default_root_dir=test_dir,
                    devices=num_gpu,
                    precision='16-mixed' if cfg.training.amp else 32,
                    accelerator='gpu',
                    logger=False,
                    strategy=DDPStrategy(find_unused_parameters=True)
                )
            else:
                test_trainer = pl.Trainer(
                    default_root_dir=test_dir,
                    devices=num_gpu,
                    precision='16-mixed' if cfg.training.amp else 32,
                    accelerator='gpu',
                    logger=False
                )
            
            # Run testing
            test_trainer.test(test_model, datamodule=datamodule, verbose=True)
            print(f"Testing completed. Results saved to: {test_dir}")
            
        except FileNotFoundError as e:
            print(f"Could not find checkpoint for testing: {e}")
            print("Skipping testing phase.")
    else:
        print("\nTraining completed. Set 'test_after_training: true' in config to run testing automatically.")

if __name__ == '__main__':
    main()
