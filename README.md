# INSTALL
```
1. python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
2. conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
3. export PYTHONPATH=.:$PYTHONPATH
```


# SUBMIT
```
python submit.py --name "actnet_zsp_mvit" --cluster "videopretrain" --node_count 1 --command "python src/task/run_tsg.py --deepspeed --deepspeed_config src/configs/ds_config.json --deepspeed_mpi --distributed --config src/configs/actnetloc_task/actnetloc_test_local.yaml --fp16" --branch main
```