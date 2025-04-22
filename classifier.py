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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

class ShipDataset(Dataset):
    def __init__(self, paths: list, label_map: dict):
        """
        metadata CSV should have columns: 'filepath', 'label'
        """
        audio_paths = []
        metadata_paths = []
        scenarios = []

        for path in paths:
            audio_paths.append(os.path.join(path, 'audio'))
            metadata_paths.append(os.path.join(path, 'metadata.csv'))
            scenarios.append(Path(path).name)

        files = []
        for i, mfile in enumerate(metadata_paths):
            mdata = pd.read_csv(mfile)
            scenario = scenarios[i]

            for label, file_index, t1norm, c1norm, p1norm, salnorm, svnorm in zip(
                mdata['label'], mdata['file_index'], mdata['t1_norm'],
                mdata['c1_norm'], mdata['p1_norm'], mdata['sal_norm'], mdata['sv_norm']
            ):
                filepath = os.path.join(audio_paths[i], str(label), f"{file_index}.wav")
                aux_data = {
                    "t1_norm": t1norm,
                    "c1_norm": c1norm,
                    "p1_norm": p1norm,
                    "sal_norm": salnorm,
                    "sv_norm": svnorm
                }
                files.append({
                    "file": filepath,
                    "label": label,
                    "aux_data": aux_data,
                    "scenario": scenario
                })
        random.shuffle(files)
        self.files = files
        self.label_map = label_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = self.files[idx]
        waveform, _ = torchaudio.load(sample['file'])
        label = self.label_map[sample['label']]
        aux_data = sample['aux_data']
        scenario = sample['scenario']
        return waveform, label, aux_data, scenario

class ShipDataModule(pl.LightningDataModule):
    def __init__(
        self, train_paths, val_paths, test_paths,
        label_map, batch_size, num_workers
    ):
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
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

class Classifier(pl.LightningModule):
    def __init__(
        self, num_outputs: int,
        frontend: Optional[torch.nn.Module] = None,
        encoder: Optional[torch.nn.Module] = None,
        cfg: dict = None,
        label_map: dict = None,
    ):
        super().__init__()
        self._frontend = frontend
        self._encoder = encoder
        self._pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

        self.use_aux = cfg["use_aux"]
        if self.use_aux:
            self.aux_fc = nn.Sequential(
                nn.Linear(5, 128),
                nn.ReLU()
            )
            head_in = 1280 + 128
        else:
            head_in = 1280

        self._head = nn.Linear(head_in, num_outputs)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = cfg["lr"]
        self.weight_decay = cfg["weight_decay"]
        self.reverse_label_map = {v: k for k, v in label_map.items()}

        # buffers
        self.train_preds, self.train_labels, self.train_scenarios = [], [], []
        self.val_preds, self.val_labels, self.val_scenarios = [], [], []
        self.test_preds, self.test_labels, self.test_scenarios = [], [], []

    def forward(self, inputs: torch.Tensor,
                aux_data: Optional[Sequence[dict]] = None):
        x = inputs
        if self._frontend is not None:
            x = self._frontend(x)
            if x.ndim == 3:
                x = x[:, None, :, :]
        if self._encoder is not None:
            x = self._encoder(x)
        x = self._pool(x)

        if self.use_aux and aux_data is not None:
            aux_list = [
                [d['t1_norm'], d['c1_norm'], d['p1_norm'], d['sal_norm'], d['sv_norm']]
                for d in aux_data
            ]
            aux_tensor = torch.tensor(aux_list,
                                      dtype=x.dtype,
                                      device=x.device)
            aux_feat = self.aux_fc(aux_tensor)
            x = torch.cat([x, aux_feat], dim=1)

        return self._head(x)

    def training_step(self, batch, batch_idx):
        audio, labels, aux_data, scenarios = batch
        logits = self(audio, aux_data)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        # log
        self.log('Train/Loss', loss, on_step=True, on_epoch=True)
        self.log('Train/Acc', acc, on_step=True, on_epoch=True)
        wandb.log({'Train Loss': loss, 'Train Acc': acc})

        # buffer
        self.train_preds.extend(preds.cpu().tolist())
        self.train_labels.extend(labels.cpu().tolist())
        self.train_scenarios.extend(scenarios)

        if self.global_step > 0 and self.global_step % 100 == 0:
            # full
            self._log_confusion_matrix(self.train_labels, self.train_preds,
                                       f"Train Confusion @ step {self.global_step}")
            self._log_metrics_table(self.train_labels, self.train_preds,
                                    self.train_scenarios,
                                    f"Train Metrics @ step {self.global_step}")
            # per scenario
            for scen in sorted(set(self.train_scenarios)):
                idxs = [i for i,s in enumerate(self.train_scenarios) if s==scen]
                t = [self.train_labels[i] for i in idxs]
                p = [self.train_preds[i] for i in idxs]
                self._log_confusion_matrix(t, p,
                                           f"Train Confusion {scen} @ step {self.global_step}")
            # clear
            self.train_preds.clear(); self.train_labels.clear(); self.train_scenarios.clear()

        return loss

    def validation_step(self, batch, batch_idx):
        audio, labels, aux_data, scenarios = batch
        logits = self(audio, aux_data)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log('Val/Loss', loss, prog_bar=True, on_epoch=True)
        self.log('Val/Acc', acc, prog_bar=True, on_epoch=True)
        wandb.log({'Val Loss': loss, 'Val Acc': acc})

        self.val_preds.extend(preds.cpu().tolist())
        self.val_labels.extend(labels.cpu().tolist())
        self.val_scenarios.extend(scenarios)
        return loss

    def validation_epoch_end(self, outputs):
        self._log_confusion_matrix(self.val_labels, self.val_preds,
                                   "Validation Confusion Matrix")
        self._log_metrics_table(self.val_labels, self.val_preds,
                                self.val_scenarios, "Validation Metrics")
        for scen in sorted(set(self.val_scenarios)):
            idxs = [i for i,s in enumerate(self.val_scenarios) if s==scen]
            t = [self.val_labels[i] for i in idxs]
            p = [self.val_preds[i] for i in idxs]
            self._log_confusion_matrix(t, p,
                                       f"Validation Confusion {scen}")
        self.val_preds.clear(); self.val_labels.clear(); self.val_scenarios.clear()

    def test_step(self, batch, batch_idx):
        audio, labels, aux_data, scenarios = batch
        logits = self(audio, aux_data)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log('Test/Loss', loss, prog_bar=True, on_epoch=True)
        self.log('Test/Acc', acc, prog_bar=True, on_epoch=True)
        wandb.log({'Test Loss': loss, 'Test Acc': acc})

        self.test_preds.extend(preds.cpu().tolist())
        self.test_labels.extend(labels.cpu().tolist())
        self.test_scenarios.extend(scenarios)
        return loss

    def test_epoch_end(self, outputs):
        self._log_confusion_matrix(self.test_labels, self.test_preds,
                                   "Test Confusion Matrix")
        self._log_metrics_table(self.test_labels, self.test_preds,
                                self.test_scenarios, "Test Metrics")
        for scen in sorted(set(self.test_scenarios)):
            idxs = [i for i,s in enumerate(self.test_scenarios) if s==scen]
            t = [self.test_labels[i] for i in idxs]
            p = [self.test_preds[i] for i in idxs]
            self._log_confusion_matrix(t, p,
                                       f"Test Confusion {scen}")
        self.test_preds.clear(); self.test_labels.clear(); self.test_scenarios.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'Val/Loss',
                    'interval': 'epoch',
                    'frequency': 1}}

    def _log_confusion_matrix(self, true, pred, title="Confusion Matrix"):
        labels_idx = sorted(self.reverse_label_map.keys())
        cm = confusion_matrix(true, pred, labels=labels_idx)
        names = [self.reverse_label_map[i] for i in labels_idx]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(names)),
            yticks=np.arange(len(names)),
            xticklabels=names,
            yticklabels=names,
            ylabel='True label',
            xlabel='Predicted label',
            title=title
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        wandb.log({title: wandb.Image(fig)})
        plt.close(fig)

    def _log_metrics_table(self, true, pred, scenarios, title="Metrics Table"):
        data = []
        for scen in sorted(set(scenarios)) + ['ALL']:
            if scen == 'ALL':
                t, p = true, pred
            else:
                idxs = [i for i,s in enumerate(scenarios) if s == scen]
                t = [true[i] for i in idxs]
                p = [pred[i] for i in idxs]
            acc = accuracy_score(t, p)
            prec, rec, f1, _ = precision_recall_fscore_support(
                t, p, average='weighted', zero_division=0)
            data.append([scen, acc, prec, rec, f1])
        table = wandb.Table(columns=['scenario','accuracy','precision','recall','f1'], data=data)
        wandb.log({title: table})
