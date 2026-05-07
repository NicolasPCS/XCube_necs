export PYTHONPATH="${PYTHONPATH}:/home/isipiran/XCube_necs"

python /home/isipiran/XCube_necs/inference/sample_shapenet_with_mirroring.py none --category plane --batch_len 5 --use_ddim
python /home/isipiran/XCube_necs/inference/sample_shapenet_with_mirroring.py none --category car --batch_len 5 --use_ddim
python /home/isipiran/XCube_necs/inference/sample_shapenet_with_mirroring.py none --category chair --batch_len 5 --use_ddim