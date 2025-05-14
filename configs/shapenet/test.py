from pathlib import Path

print("Estoy en:", Path.cwd())
print("Existe?", Path("../../data/shapenet").resolve().exists())