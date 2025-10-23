# data/datamodule.py

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from data.citywalk_dataset import CityWalkSampler
from data.citywalk_feat_dataset import CityWalkFeatDataset
from torch.utils.data.distributed import DistributedSampler


def citywalk_collate(batch):
    """Custom collate function that avoids resizing non-resizable storages."""

    def _collate_recursive(items):
        template = next((item for item in items if item is not None), None)
        if template is None:
            return None

        if isinstance(template, torch.Tensor):
            filled = []
            for item in items:
                if item is None:
                    filled.append(torch.zeros_like(template))
                else:
                    filled.append(item.clone())
            return torch.stack(filled, dim=0)

        if isinstance(template, (float, int)):
            dtype = torch.float32 if isinstance(template, float) else torch.long
            values = [0 if item is None else item for item in items]
            return torch.tensor(values, dtype=dtype)

        if isinstance(template, dict):
            keys = set()
            for item in items:
                if isinstance(item, dict):
                    keys.update(item.keys())
            return {
                key: _collate_recursive(
                    [
                        item.get(key) if isinstance(item, dict) else None
                        for item in items
                    ]
                )
                for key in keys
            }

        # Fallback: keep raw objects (e.g., strings) as lists
        return items

    return _collate_recursive(batch)


class CityWalkFeatDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CityWalkFeatDataset(self.cfg, mode='train')
            self.val_dataset = CityWalkFeatDataset(self.cfg, mode='val')

        if stage == 'test' or stage is None:
            self.test_dataset = CityWalkFeatDataset(self.cfg, mode='test')

    def train_dataloader(self):
        # Use DistributedSampler for multi-GPU training, fallback to custom sampler for single GPU
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            sampler = CityWalkSampler(self.train_dataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=citywalk_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=citywalk_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=citywalk_collate,
        )
