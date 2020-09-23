#!/usr/bin/env bash
python scripts/train_carla_van_horizon_vpz.py
    --dataset_dir=<train.tf_records>
    --train_dir=<experiment_dir>
    --checkpoint_path=<vgg_16.ckpt>
    --checkpoint_exclude_scopes=vgg_16/fc8
    --model_name=vgg-16
    --save_summaries_secs=30
    --save_interval_secs=300
    --learning_rate=0.01
    --optimizer=adam
    --batch_size=8