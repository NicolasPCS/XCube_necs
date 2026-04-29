import torch

path = "/home/isipiran/XCube_necs/data/XCube_DatasetV2/128/02691156/1a04e3eab45ca15dd86060f189eb133.pkl"

obj = torch.load(path, map_location="cpu")

print("Tipo del objeto:", type(obj))
#print("Atributos disponibles:", dir(obj))
print(obj.keys())

points = obj["points"]
normals = obj["normals"]

print("=== POINTS ===")
print("Tipo:", type(points))
print("Shape:", points.shape if hasattr(points, "shape") else "no tensor")
print("Ejemplo de 5 puntos:\n", points[:5])

print([m for m in dir(points) if not m.startswith("_")])

print("\n=== NORMALS ===")
print("Tipo:", type(normals))
print("Shape:", normals.shape if hasattr(normals, "shape") else "no tensor")
print("Ejemplo de 5 normales:\n", normals[:5])
