python tools/run_net.py \
    --cfg configs/Kinetics/MVIT_B_32x3_CONV.yaml \
    DATA.PATH_TO_DATA_DIR data/kinetics400 \
    TEST.CHECKPOINT_FILE_PATH model_zoo/mvit/K400_MVIT_B_32x3_CONV.pyth \
    TRAIN.ENABLE False 