python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="200.1.17.169" --master_port=1234  train_stylegan2-new.py --path /home/jenny2/HypStyleGAN/lsun_cat/thfeee_eeee_c_1e-3 --cfg thfeeeeeee --c 0.001 --r1 1 --lr 0.0005 --distributed 1 --dataset 1 --seed 1 --size 256 --iter 450001 --batch_size 16 --ckpt /home/jenny2/HypStyleGAN/lsun_cat/thfeee_eeee_c_1e-3/checkpoint/030000.pt