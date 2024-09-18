# optimize.py
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
import os

from util import *
import render
import loss
import imageio
import sys
sys.path.append('..')
from flexicubes1 import FlexiCubes, FeatureRecognitionNet  # 确保从正确位置导入
from gans import Generator  # 确保从正确位置导入
import trimesh

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.

# 加载特征识别模型
feature_model = FeatureRecognitionNet()
feature_model.load_state_dict(torch.load('./models/feature_recognition_net.pth'))
feature_model.eval()

# 加载生成对抗网络模型
generator = Generator()
generator.load_state_dict(torch.load('./models/generator.pth'))
generator.eval()

def extract_features_from_mesh(mesh):
    vertices = mesh.vertices
    normals = mesh.normals
    curvatures = compute_curvatures(mesh)
    features = torch.cat((vertices, normals, curvatures), dim=1)
    return features

def compute_curvatures(mesh):
    v0 = mesh.vertices[mesh.faces[:, 0], :]
    v1 = mesh.vertices[mesh.faces[:, 1], :]
    v2 = mesh.vertices[mesh.faces[:, 2], :]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    curvatures = torch.norm(face_normals, dim=1, keepdim=True)
    return curvatures

def apply_refinement_to_mesh(mesh, refined_mesh):
    mesh.vertices = refined_mesh.vertices
    mesh.faces = refined_mesh.faces
    mesh.compute_normals()  # 重新计算法线
    return mesh


def optimize_mesh(mesh):
    # 提取网格特征
    features = extract_features_from_mesh(mesh)
    # 使用特征识别模型判断是否需要优化
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    need_refinement = feature_model(features_tensor)

    # 根据模型输出决定是否进行优化
    if need_refinement.item() > 0.5:
        # 使用GAN生成新的网格数据
        noise = torch.randn((1, 100, 1, 1))  # 噪声维度需根据GAN模型调整
        refined_mesh_data = generator(noise)

        # 假设generator输出与mesh格式兼容（需要您根据实际情况调整）
        refined_vertices = refined_mesh_data['vertices']
        refined_faces = refined_mesh_data['faces']

        # 更新网格数据
        mesh.vertices = refined_vertices
        mesh.faces = refined_faces
        mesh.auto_normals()  # 重新计算法线

    return mesh


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.normals = self.calculate_normals()

    def calculate_normals(self):
        # 计算法线的逻辑
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        # 使用叉积计算面法线
        normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        normals = normals / torch.norm(normals, dim=1, keepdim=True)  # 标准化法线
        return normals

    def auto_normals(self):
        # 确保调用此函数以更新或重新计算法线
        self.normals = self.calculate_normals()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='flexicubes optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-i', '--iter', type=int, default=1000)
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('-r', '--train_res', nargs=2, type=int, default=[2048, 2048])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--voxel_grid_res', type=int, default=64)
    parser.add_argument('--sdf_loss', type=bool, default=True)
    parser.add_argument('--develop_reg', type=bool, default=False)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)
    parser.add_argument('-dr', '--display_res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-si', '--save_interval', type=int, default=20)
    FLAGS = parser.parse_args()
    device = 'cuda'

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()

    # Load GT mesh
    gt_mesh = load_mesh(FLAGS.ref_mesh, device)
    gt_mesh.auto_normals()  # compute face normals for visualization

    # 确保在调用 optimize_mesh 前网格已正确初始化
    # Create and initialize FlexiCubes
    fc = FlexiCubes(device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(FLAGS.voxel_grid_res)
    x_nx3 *= 2  # scale up the grid so that it's larger than the target object

    vertices, faces, L_dev = fc(x_nx3, None, cube_fx8, FLAGS.voxel_grid_res)

    flexicubes_mesh = Mesh(vertices, faces)
    flexicubes_mesh.auto_normals()  # 确保法线计算正确

    sdf = torch.rand_like(x_nx3[:, 0]) - 0.1  # randomly init SDF
    sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    # set per-cube learnable weights to zeros
    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda')
    weight = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

    # Retrieve all the edges of the voxel grid; these edges wimmmtilized to
    # compute the regularization loss in subsequent steps of the process.
    all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
    grid_edges = torch.unique(all_edges, dim=0)

    # Setup optimizer
    optimizer = torch.optim.Adam([sdf, weight, deform], lr=FLAGS.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x))

    # Train loop
    for it in range(FLAGS.iter):
        optimizer.zero_grad()
        # sample random camera poses
        mv, mvp = render.get_random_camera_batch(FLAGS.batch, iter_res=FLAGS.train_res, device=device, use_kaolin=False)
        # render gt mesh
        target = render.render_mesh_paper(gt_mesh, mv, mvp, FLAGS.train_res)
        # extract and render FlexiCubes mesh
        grid_verts = x_nx3 + (2 - 1e-8) / (FLAGS.voxel_grid_res * 2) * torch.tanh(deform)
        vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res,
                                    beta_fx12=weight[:, :12], alpha_fx8=weight[:, 12:20],
                                    gamma_f=weight[:, 20], training=True)
        flexicubes_mesh = Mesh(vertices, faces)
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, FLAGS.train_res)

        # evaluate reconstruction loss
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        depth_loss = (((((buffers['depth'] - (target['depth'])) * target['mask']) ** 2).sum(-1) + 1e-8)).sqrt().mean() * 10

        t_iter = it / FLAGS.iter
        sdf_weight = FLAGS.sdf_regularizer - (FLAGS.sdf_regularizer - FLAGS.sdf_regularizer / 20) * min(1.0, 4.0 * t_iter)
        reg_loss = loss.sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight  # Loss to eliminate internal floaters that are not visible
        reg_loss += L_dev.mean() * 0.5
        reg_loss += (weight[:, :20]).abs().mean() * 0.1
        total_loss = mask_loss + depth_loss + reg_loss

        if FLAGS.sdf_loss:  # optionally add SDF loss to eliminate internal structures
            with torch.no_grad():
                pts = sample_random_points(1000, gt_mesh)
                gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
            pred_sdf = compute_sdf(pts, flexicubes_mesh.vertices, flexicubes_mesh.faces)
            total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3

        # optionally add developability regularizer, as described in paper section 5.2
        if FLAGS.develop_reg:
            reg_weight = max(0, t_iter - 0.8) * 5
            if reg_weight > 0:  # only applied after shape converges
                reg_loss = loss.mesh_developable_reg(flexicubes_mesh).mean() * 10
                reg_loss += (deform).abs().mean()
                reg_loss += (weight[:, :20]).abs().mean()
                total_loss = mask_loss + depth_loss + reg_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Use feature recognition and GANs for mesh refinement
        flexicubes_mesh = optimize_mesh(flexicubes_mesh)

        if (it % FLAGS.save_interval == 0 or it == (FLAGS.iter - 1)):  # save normal image for visualization
            with torch.no_grad():
                # extract mesh with training=False
                vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res,
                                            beta_fx12=weight[:, :12], alpha_fx8=weight[:, 12:20],
                                            gamma_f=weight[:, 20], training=False)
                flexicubes_mesh = Mesh(vertices, faces)

                flexicubes_mesh.auto_normals()  # compute face normals for visualization
                mv, mvp = render.get_rotate_camera(it // FLAGS.save_interval, iter_res=FLAGS.display_res, device=device, use_kaolin=False)
                val_buffers = render.render_mesh_paper(flexicubes_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res,
                                                       return_types=["normal"], white_bg=True)
                val_image = ((val_buffers["normal"][0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

                gt_buffers = render.render_mesh_paper(gt_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res,
                                                      return_types=["normal"], white_bg=True)
                gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}.png'.format(it)), np.concatenate([val_image, gt_image], 1))
                print(f"Optimization Step [{it}/{FLAGS.iter}], Loss: {total_loss.item():.4f}")

    # Save output
    mesh_np = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    mesh_np.export(os.path.join(FLAGS.out_dir, 'output_mesh.obj'))
