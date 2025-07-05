import open3d as o3d
import numpy as np
import time


# 生成示例点云（实际使用时替换为你的点云数据）
def generate_sample_point_cloud():
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    return mesh.sample_points_poisson_disk(number_of_points=1000)


# 点云重建函数
def reconstruct_point_cloud(point_cloud):
    # 估计法向量（Poisson重建必需）
    point_cloud.estimate_normals()

    # 执行Poisson重建
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=9  # 控制重建细节程度（8-12之间）
    )
    return mesh


# 主流程
start_time = time.time()

# 1. 加载点云（替换为你的点云路径）
pcd = o3d.io.read_point_cloud("2025_05_09_13_30_31.ply")
pcd = generate_sample_point_cloud()  # 使用示例数据

# 2. 执行重建
reconstructed_mesh = reconstruct_point_cloud(pcd)

# 3. 保存结果
o3d.io.write_triangle_mesh("reconstructed_mesh.ply", reconstructed_mesh)

print(f"重建完成! 耗时: {time.time() - start_time:.2f}秒")