import torch

# Ruta a tu checkpoint
checkpoint_path = "/home/ncaytuir/data-local/wandb/xcube-shapenet/y98joza4/checkpoints/epoch=000099-step=000009500.ckpt"

# Carga el checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Imprime los hiperparámetros guardados
#print("Hiperparámetros guardados en el checkpoint:")
#print(checkpoint['hyper_parameters'])

# Opcional: Imprime las claves de las capas para verificar los nombres
# print(checkpoint['state_dict'].keys())

w = checkpoint["state_dict"]["unet.pre_kl_bottleneck.pre_kl_bottleneck_1.SingleConv1.Conv.weight"]
print("Checkpoint weight shape:", w.shape)

for k,v in checkpoint["hyper_parameters"].items():
    print(k,":",v)


""" {'exec': None, 'include': None, 'test_set_shuffle': False, 'batch_size': 32, 'accumulate_grad_batches': 1, 'visualize': False, 'name': 'shapenet/plane_VAE_dense', 'model': 'autoencoder', 'tree_depth': 4, 'voxel_size': [0.01, 0.01, 0.01], 'resolution': 128, 'use_fvdb_loader': True, 'use_hash_tree': True, 'use_input_normal': True, 'use_input_semantic': False, 'use_input_intensity': False, 'cut_ratio': 16, 'kl_weight': 0.03, 'normalize_kld': True, 'enable_anneal': False, 'kl_weight_min': 1e-07, 'kl_weight_max': 0.03, 'anneal_star_iter': 0, 'anneal_end_iter': 70000, 'supervision': {'structure_weight': 20.0, 'normal_weight': 300.0, 'color_weight': 0.0, 'semantic_weight': 0.0}, 'optimizer': 'Adam', 'learning_rate': {'init': 0.0001, 'decay_mult': 0.7, 'decay_step': 50000, 'clip': 1e-06}, 'weight_decay': 0.0, 'grad_clip': 0.5, 'network': 
 
 {'encoder': {'c_dim': 32}, 'unet': {'target': 'StructPredictionNet', 'params': {'in_channels': 32, 'num_blocks': '${tree_depth}', 'f_maps': 64, 'neck_dense_type': 'HAND_CRAFTED', 'neck_bound': [8, 8, 8], 'num_res_blocks': 1, 'use_residual': False, 'order': 'gcr', 'is_add_dec': False, 'use_attention': False, 'use_checkpoint': False}}}, 
 
 '_shapenet_path': '/home/ncaytuir/data-local/XCube_necs/data/XCube_DatasetV2', '_shapenet_categories': ['02691156'], '_shapenet_custom_name': 'shapenet', 'train_dataset': 'ShapeNetDataset', 'train_val_num_workers': 16, 'train_kwargs': {'onet_base_path': '${_shapenet_path}', 'resolution': '${resolution}', 'categories': '${_shapenet_categories}', 'custom_name': '${_shapenet_custom_name}', 'split': 'train', 'random_seed': 0}, 'val_dataset': 'ShapeNetDataset', 'val_kwargs': {'onet_base_path': '${_shapenet_path}', 'resolution': '${resolution}', 'categories': '${_shapenet_categories}', 'custom_name': '${_shapenet_custom_name}', 'split': 'val', 'random_seed': 'fixed'}, 'test_dataset': 'ShapeNetDataset', 'test_num_workers': 8, 'test_kwargs': {'onet_base_path': '${_shapenet_path}', 'resolution': '${resolution}', 'categories': '${_shapenet_categories}', 'custom_name': '${_shapenet_custom_name}', 'split': 'test', 'random_seed': 'fixed'}, 'remain_h': False, 'pretrained_weight': None, 'use_input_color': False, 'with_color_branch': False, 'with_normal_branch': True, 'with_semantic_branch': False} """