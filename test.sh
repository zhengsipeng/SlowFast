python tools/run_net.py \
    --mode test \
    --cfg configs/Kinetics/MVIT_B_32x3_CONV.yaml \
    --deepspeed \
    --deepspeed_config configs/ds_cfg.json \
    --deepspeed_mpi \
    --distributed \
    --fp16 \
    