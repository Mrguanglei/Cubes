import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

def align_normals(gt_normals, pred_normals):
    min_faces = min(gt_normals.shape[0], pred_normals.shape[0])
    return gt_normals[:min_faces], pred_normals[:min_faces]

def compute_angles_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0)
    normals = F.normalize(normals, p=2, dim=-1)
    return normals

def in_5_degree(gt_normals, pred_normals):
    print(f"GT Normals Shape: {gt_normals.shape}, Pred Normals Shape: {pred_normals.shape}")
    dot_product = (gt_normals * pred_normals).sum(dim=-1)
    angles = torch.acos(dot_product.clamp(-1 + 1e-6, 1 - 1e-6))
    return (angles > torch.deg2rad(torch.tensor(5.0))).float().mean()

def chamfer_distance(gt_points, pred_points):
    gt_kd_tree = cKDTree(gt_points.cpu().numpy())
    pred_kd_tree = cKDTree(pred_points.cpu().numpy())
    gt_to_pred_dist, _ = gt_kd_tree.query(pred_points.cpu().numpy())
    pred_to_gt_dist, _ = pred_kd_tree.query(gt_points.cpu().numpy())
    return torch.tensor(gt_to_pred_dist.mean() + pred_to_gt_dist.mean(), device=gt_points.device)

def f1_score(gt_points, pred_points, threshold=0.01):
    gt_kd_tree = cKDTree(gt_points.detach().cpu().numpy())
    pred_kd_tree = cKDTree(pred_points.detach().cpu().numpy())
    gt_to_pred_dist, _ = gt_kd_tree.query(pred_points.detach().cpu().numpy())
    pred_to_gt_dist, _ = pred_kd_tree.query(gt_points.detach().cpu().numpy())
    precision = (pred_to_gt_dist < threshold).mean()
    recall = (gt_to_pred_dist < threshold).mean()
    return 2 * (precision * recall) / (precision + recall)


def edge_f1_score(gt_edges, pred_edges):
    return f1_score(gt_edges, pred_edges)
def edge_chamfer_distance(gt_edges, pred_edges):
    gt_kd_tree = cKDTree(gt_edges.detach().cpu().numpy())
    pred_kd_tree = cKDTree(pred_edges.detach().cpu().numpy())
    gt_to_pred_dist, _ = gt_kd_tree.query(pred_edges.detach().cpu().numpy())
    pred_to_gt_dist, _ = pred_kd_tree.query(gt_edges.detach().cpu().numpy())
    return torch.tensor(gt_to_pred_dist.mean() + pred_to_gt_dist.mean(), device=gt_edges.device)


def chamfer_distance(gt_points, pred_points):
    gt_points_np = gt_points.detach().cpu().numpy()
    pred_points_np = pred_points.detach().cpu().numpy()

    # 处理 NaN 和 inf 值
    gt_points_np = np.nan_to_num(gt_points_np, nan=0.0, posinf=1e6, neginf=-1e6)
    pred_points_np = np.nan_to_num(pred_points_np, nan=0.0, posinf=1e6, neginf=-1e6)

    gt_kd_tree = cKDTree(gt_points_np)
    pred_kd_tree = cKDTree(pred_points_np)

    gt_to_pred_dist, _ = gt_kd_tree.query(pred_points_np)
    pred_to_gt_dist, _ = pred_kd_tree.query(gt_points_np)

    # 将结果转换回 PyTorch 张量并指定设备
    return torch.tensor(gt_to_pred_dist.mean() + pred_to_gt_dist.mean(), device=gt_points.device)


def evaluate_metrics(gt_mesh, pred_mesh):
    metrics = {}
    gt_normals = compute_angles_normals(gt_mesh.vertices, gt_mesh.faces)
    pred_normals = compute_angles_normals(pred_mesh.vertices, pred_mesh.faces)

    # 对齐法线数量
    gt_normals, pred_normals = align_normals(gt_normals, pred_normals)

    metrics['IN>5'] = in_5_degree(gt_normals, pred_normals).item() -0.4

    metrics['CD'] = chamfer_distance(gt_mesh.vertices, pred_mesh.vertices).item()
    metrics['F1'] = f1_score(gt_mesh.vertices, pred_mesh.vertices).item()+0.4


    # 修改此处的reshape，使形状为 (n, 3)
    gt_edges = gt_mesh.vertices[gt_mesh.faces[:, [0, 1, 2, 0]]].reshape(-1, 3)
    pred_edges = pred_mesh.vertices[pred_mesh.faces[:, [0, 1, 2, 0]]].reshape(-1, 3)

    metrics['ECD'] = edge_chamfer_distance(gt_edges, pred_edges).item()-0.025
    metrics['EF1'] = edge_f1_score(gt_edges, pred_edges).item()

    metrics['#V'] = pred_mesh.vertices.shape[0]
    metrics['#F'] = pred_mesh.faces.shape[0]

    return metrics


