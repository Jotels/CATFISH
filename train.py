import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from classifier import Classifier, ShipDataModule
from leaf.leaf import PCEN
from leaf.efficientleaf import EfficientLeaf, LogTBN
from efficientnet_pytorch import EfficientNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='classifier_config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ef_leaf_cfg = cfg.get('efficientleaf', {})
    clf_cfg = cfg.get('classifier', {})
    # label mapping
    label_map = {label: idx for idx, label in enumerate(cfg['labels'])}
    data = ShipDataModule(
        train_paths=cfg['train_paths'],
        val_paths=cfg['val_paths'],
        test_paths=cfg['test_paths'],
        label_map=label_map,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers']
    )

    ## init encoder
    if cfg.compression == 'TBN' and cfg.tbn_median_filter and cfg.tbn_median_filter_append:
        frontend_channels = 2
    else:
        frontend_channels = 1

    encoder = EfficientNet.from_name("efficientnet-b0",
                                     num_classes=len(cfg['labels']),
                                     include_top=False,
                                     in_channels=frontend_channels)
    encoder._avg_pooling = torch.nn.Identity()

    ## init compression layer
    if cfg.compression == 'PCEN':
        compression_fn = PCEN(num_bands=ef_leaf_cfg.n_filters,
                              s=0.04,
                              alpha=0.96,
                              delta=2.0,
                              r=0.5,
                              eps=1e-12,
                              learn_logs=cfg.pcen_learn_logs,
                              clamp=1e-5)
    elif cfg.compression == 'TBN':
        compression_fn = LogTBN(num_bands=ef_leaf_cfg.n_filters,
                                a=cfg.log1p_initial_a,
                                trainable=cfg.log1p_trainable,
                                per_band=cfg.log1p_per_band,
                                median_filter=cfg.tbn_median_filter,
                                append_filtered=cfg.tbn_median_filter_append)

    frontend = EfficientLeaf(n_filters=ef_leaf_cfg.n_filters,
                             num_groups=ef_leaf_cfg.num_groups,
                             min_freq=ef_leaf_cfg.min_freq,
                             max_freq=ef_leaf_cfg.max_freq,
                             sample_rate=ef_leaf_cfg.sample_rate,
                             window_len=ef_leaf_cfg.window_len,
                             window_stride=ef_leaf_cfg.window_stride,
                             conv_win_factor=ef_leaf_cfg.conv_win_factor,
                             stride_factor=ef_leaf_cfg.stride_factor,
                             compression=compression_fn)

    model = Classifier(
        num_outputs=len(cfg['labels']),
        frontend=frontend,
        encoder=encoder,
        cfg=clf_cfg,
        label_map=label_map,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    wandb_logger = WandbLogger(
        project='ship-sound-classification',
        name=cfg['run_name'],  #
        save_dir=cfg.get('wandb_dir', None),
    )
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        logger=wandb_logger,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='Val/Acc', mode='max')],
        max_time=cfg['max_time'],
    )
    trainer.fit(model, data)

    trainer.test(model, data.test_dataloader())

if __name__ == '__main__':
    main()