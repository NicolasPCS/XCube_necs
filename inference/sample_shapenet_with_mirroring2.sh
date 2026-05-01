export PYTHONPATH=$PYTHONPATH:.

#python /home/ncaytuir/data-local/XCube_necs/inference/sample_shapenet_with_mirroring.py none --category plane --batch_len 4 --ema --use_ddim --original
python /home/ncaytuir/data-local/XCube_necs/inference/sample_shapenet_with_mirroring.py none --category car --batch_len 4 --ema --use_ddim --original
python /home/ncaytuir/data-local/XCube_necs/inference/sample_shapenet_with_mirroring.py none --category chair --batch_len 2 --ema --use_ddim --original