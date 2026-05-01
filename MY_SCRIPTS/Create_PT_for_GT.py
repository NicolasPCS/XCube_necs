import os
import json
import torch
import numpy as np

input_path = "/home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/eval_408objects/complete_point_clouds"
output_path = "/home/ncaytuir/data-local/exp/0604/airplane/7807deh_train_lion_B10/eval_408objects/complete_point_clouds/complete_generated_shapes.pt"

# Listar y ordenar archivos .npy
file_list = sorted([f for f in os.listdir(input_path) if f.endswith(".npy")]) #[:250] # 250 culd be adjusted

all_pcs = []

for filename in file_list:
    file_path = os.path.join(input_path, filename)
    pc = np.load(file_path)

    print(f"La nube tenia {pc.shape} ahora tiene {pc.shape}")

    all_pcs.append(pc)

# Convetir a tensor
ref = torch.from_numpy(np.stack(all_pcs)).float() # (N, 2048, 3)
mean = ref.mean(dim=1, keepdim=True) # (N, 1, 3)
std = ref.std(dim=1, keepdim=True).mean(dim=2, keepdim=True) # (N, 1, 1)

# Save pt
torch.save({'ref': ref, 'mean': mean, 'std': std}, output_path)
print(f"Guardado {ref.shape[0]} nubes en {output_path}")