#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#import warnings
#warnings.filterwarnings("ignore")
"""Wrapper to train and test a video classification model."""
import deepspeed
from src.config.defaults import assert_and_infer_cfg
from src.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    
    # Perform training.
    if args.mode == "train":
        train(args, cfg)
    elif args.mode == "test":
        # Perform multi-clip testing.
        test(args, cfg)
    elif args.mode == "vis":
        raise NotImplementedError
        # Perform model visualization.
        '''
        if cfg.TENSORBOARD.ENABLE and (
            cfg.TENSORBOARD.MODEL_VIS.ENABLE
            or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
        ):
            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)
        '''
    else:
        # Run demo.
        demo(cfg)


if __name__ == "__main__":
    deepspeed.init_distributed()
    main()
