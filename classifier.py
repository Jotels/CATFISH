import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from typing import Any, Optional, Sequence, Tuple
import wandb
import random

class ShipDataset(Dataset):
    def __init__(self, paths: list, label_map: dict):
        """
        metadata CSV should have columns: 'filepath', 'label'
        """
        import pandas as pd
        audio_paths = []
        metadata_paths = []

        for path in paths:
            audio_paths.append(os.path.join(path, 'audio'))
            metadata_paths.append(os.path.join(path, 'metadata.csv'))


        files = []
        for i, mfile in enumerate(metadata_paths):
            mdata = pd.read_csv(mfile)

            for label, file_index, t1norm, c1norm, p1norm, salnorm, svnorm in zip(mdata['label'], mdata['file_index'], mdata['t1_norm'], mdata['c1_norm'], mdata['p1_norm'], mdata['sal_norm'], mdata['sv_norm']):
                filepath = os.path.join(audio_paths[file_index], str(label), f"{file_index}.wav")
                aux_data = {"t1_norm": t1norm, "c1_norm": c1norm, "p1_norm": p1norm, "sal_norm": salnorm, "sv_norm": svnorm}
                files.append({"file": filepath,
                              "label": label,
                              "aux_data": aux_data})
        self.files = files
        random.shuffle(self.files)
        self.label_map = label_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = self.files[idx]
        audiofile = sample['file']
        waveform, _ = torchaudio.load(audiofile)
        label = self.label_map[sample['class']]
        aux_data = sample['aux_data']
        return waveform, label, aux_data


class ShipDataModule(pl.LightningDataModule):
    def __init__(self, train_paths,
                 val_paths,
                 test_paths,
                 sample_rate,
                 duration,
                 label_map,
                 batch_size,
                 num_workers):
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.label_map = label_map
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = ShipDataset(self.train_paths, self.label_map)
        self.val_ds = ShipDataset(self.val_paths, self.label_map)
        self.test_ds = ShipDataset(self.test_paths, self.label_map)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class Classifier(pl.LightningModule):
    def __init__(
        self,
        num_outputs: int,
        frontend: Optional[torch.nn.Module] = None,
        encoder: Optional[torch.nn.Module] = None,
        cfg: dict = None
    ):
        super().__init__()
        self._frontend = frontend
        self._encoder = encoder
        self._pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

        # whether to use the 5‑dim aux vector
        self.use_aux = cfg.use_aux
        if self.use_aux:
            # project 5‑d aux → 128
            self.aux_fc = nn.Sequential(
                nn.Linear(5, 128),
                nn.ReLU()
            )
            head_in = 1280 + 128
        else:
            head_in = 1280

        self._head = nn.Linear(head_in, num_outputs)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # hyperparams
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay

    def forward(self, inputs: torch.Tensor, aux_data: Optional[Sequence[dict]] = None):
        x = inputs
        if self._frontend is not None:
            x = self._frontend(x)
            if x.ndim == 3:
                x = x[:, None, :, :]  # add channel dim
        if self._encoder is not None:
            x = self._encoder(x)

        x = self._pool(x)  # B × 1280

        if self.use_aux and aux_data is not None:
            # aux_data is a list of dicts with keys t1_norm, c1_norm, p1_norm, sal_norm, sv_norm
            aux_list = [
                [d['t1_norm'], d['c1_norm'], d['p1_norm'], d['sal_norm'], d['sv_norm']]
                for d in aux_data
            ]
            aux_tensor = torch.tensor(aux_list, dtype=x.dtype, device=x.device)
            aux_feat = self.aux_fc(aux_tensor)  # B × 128
            x = torch.cat([x, aux_feat], dim=1)    # B × (1280+128)

        return self._head(x)

    def training_step(self, batch, batch_idx):
        audio, labels, aux_data = batch
        logits = self(audio, aux_data)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log('Train/Loss', loss, on_step=True, on_epoch=True)
        self.log('Train/Acc', acc, on_step=True, on_epoch=True)
        wandb.log({'Train Loss': loss, 'Train Acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        audio, labels, aux_data = batch
        logits = self(audio, aux_data)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log('Val/Loss', loss, prog_bar=True, on_epoch=True)
        self.log('Val/Acc', acc, prog_bar=True, on_epoch=True)
        wandb.log({'Val Loss': loss, 'Val Acc': acc})
        return loss

    def test_step(self, batch, batch_idx):
        audio, labels, aux_data = batch
        logits = self(audio, aux_data)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log('Test/Loss', loss, prog_bar=True, on_epoch=True)
        self.log('Test/Acc', acc, prog_bar=True, on_epoch=True)
        wandb.log({'Test Loss': loss, 'Test Acc': acc})
        return loss

    def configure_optimizers(self):
        # AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Reduce LR on plateau of Val/Loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'Val/Loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
