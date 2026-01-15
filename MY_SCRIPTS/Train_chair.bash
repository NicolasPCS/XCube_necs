#COARSE
python train.py ./configs/shapenet/chair/train_diffusion_16x16x16_dense.yaml --wname 16x16x16_kld-0.03 --eval_interval 5 --gpus 3 --batch_size 1 --accumulate_grad_batches 32

#FINE
#python train.py ./configs/shapenet/plane/train_diffusion_128x128x128_sparse.yaml --wname 128x128x128_kld-1.0_normal_cond --eval_interval 5 --gpus 3 --batch_size 8 --accumulate_grad_batches 8 --save_topk 2 --save_every 30