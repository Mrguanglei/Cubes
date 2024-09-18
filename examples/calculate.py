import torch

# 计算IN>5°（法线角度差超过5°的比例）
def calculate_in_5(normal1, normal2):
    cos_theta = torch.nn.functional.cosine_similarity(normal1, normal2)
    angle_diff = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # 计算角度差
    in_5_deg = torch.sum(angle_diff > (5.0 * torch.pi / 180)) / normal1.shape[0]
    return in_5_deg.item()

# 计算倒角距离（Chamfer Distance）
def calculate_cd(pointcloud1, pointcloud2):
    dist1, _ = torch.min(torch.cdist(pointcloud1, pointcloud2), dim=1)
    dist2, _ = torch.min(torch.cdist(pointcloud2, pointcloud1), dim=1)
    cd = (torch.mean(dist1) + torch.mean(dist2)) / 2.0
    return cd.item()

# 计算F1分数
def calculate_f1(cd, threshold):
    return (cd < threshold).float().mean().item()

# 计算边缘倒角距离（Edge Chamfer Distance）
def calculate_ecd(edges1, edges2):
    return calculate_cd(edges1, edges2)

# 计算边缘F1分数（Edge F1 Score）
def calculate_ef1(ecd, threshold):
    return calculate_f1(ecd, threshold)

# 计算顶点数和面数
def count_vertices_faces(mesh):
    num_vertices = mesh.vertices.shape[0]
    num_faces = mesh.faces.shape[0]
    return num_vertices, num_faces
