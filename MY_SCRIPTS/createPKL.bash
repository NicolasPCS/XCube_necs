
# COARSE LEVEL
## CAR
#python CreatePKLFiles.py /home/ncaytuir/data/Datasets/ShapeNetCore.v6.PC15k/02958343 /home/isipiran/XCube_necs/data/XCube_DatasetV2/128/02958343 True
## CHAIR
#python CreatePKLFiles.py /home/ncaytuir/data/Datasets/ShapeNetCore.v6.PC15k/03001627 /home/isipiran/XCube_necs/data/XCube_DatasetV2/128/03001627 True

# FINE LEVER
## AIRPLANE
python MY_SCRIPTS/CreatePKLFiles.py /home/ncaytuir/data/Datasets/ShapeNetCore.v6.PC15k/02691156 /home/isipiran/XCube_necs/data/XCube_DatasetV2/512/02691156 False

## CAR
python MY_SCRIPTS/CreatePKLFiles.py /home/ncaytuir/data/Datasets/ShapeNetCore.v6.PC15k/02958343 /home/isipiran/XCube_necs/data/XCube_DatasetV2/512/02958343 False

## CHAIR
python MY_SCRIPTS/CreatePKLFiles.py /home/ncaytuir/data/Datasets/ShapeNetCore.v6.PC15k/03001627 /home/isipiran/XCube_necs/data/XCube_DatasetV2/512/03001627 False

cd /home/isipiran/XCube_necs

## DELETE THIS LATER
python train.py ./configs/shapenet/plane/train_vae_128x128x128_sparse.yaml --wname 512_to_128-kld-1.0 --max_epochs 100 --gpus 3 --batch_size 8 --accumulate_grad_batches 2

python train.py ./configs/shapenet/plane/train_diffusion_128x128x128_sparse.yaml --wname 128x128x128_kld-1.0_normal_cond --eval_interval 5 --gpus 3 --batch_size 8 --accumulate_grad_batches 8 --save_topk 2 --save_every 30