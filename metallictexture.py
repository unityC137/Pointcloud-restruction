import open3d as o3d
import numpy as np
import cupy as cp

# ---------------------- GPU初始化配置 ----------------------
o3d.core.Device("CUDA:0")  # 指定使用第一个CUDA设备[8](@ref)

# --------------------------- 加载点云 ---------------------------
pcd = o3d.t.io.read_point_cloud("").to(o3d.core.Device("CUDA:0"))  # 数据加载到GPU
if pcd.is_empty():
    raise ValueError("点云为空，请检查文件路径")

print(f"点云点数: {pcd.point.positions.shape[0]}")

# ---------------------- GPU法线估计加速 ----------------------
print("->正在估计法向量...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.01, max_nn=30), device=o3d.core.Device("CUDA:0"))  # GPU加速法线估计[8](@ref)

# ---------------------- GPU泊松重建加速 ----------------------
print("->开始泊松重建...")
try:
    # 转换到Tensor格式以支持GPU运算
    pcd_tensor = o3d.t.geometry.PointCloud.from_legacy(pcd).to(o3d.core.Device("CUDA:0"))

    # 启用GPU加速的泊松重建
    mesh = o3d.t.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_tensor,
        depth=10,
        scale=1.1,
        device=o3d.core.Device("CUDA:0"),  # 指定GPU设备[8](@ref)
        n_threads=8  # 多线程优化
    )[0].to_legacy()  # 转回Legacy格式

    if len(mesh.vertices) == 0:
        raise RuntimeError("泊松重建失败，尝试调整参数")

    print(f"重建成功，顶点数: {len(mesh.vertices)}")

    # ---------------------- GPU网格优化 ----------------------
    # 创建GPU加速的网格对象
    mesh_gpu = o3d.t.geometry.TriangleMesh.from_legacy(mesh).to(o3d.core.Device("CUDA:0"))

    # 并行化网格优化操作
    mesh_gpu = mesh_gpu.remove_degenerate_triangles()
    mesh_gpu = mesh_gpu.remove_duplicated_vertices()
    mesh_gpu = mesh_gpu.remove_non_manifold_edges()

    mesh = mesh_gpu.to_legacy()  # 转回CPU格式用于可视化

    # ------------------------- 旧版可视化方法 ---------------------------
    # 旧版API不支持point_size和mesh_show_back_face参数
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800)

    # 添加点云和网格
    vis.add_geometry(pcd)
    vis.add_geometry(mesh)

    # 手动设置渲染选项（旧版方法）
    opt = vis.get_render_option()
    opt.point_size = 0.03  # 设置点大小
    opt.mesh_show_back_face = True  # 显示背面
    opt.background_color = np.array([0, 0, 0])  # 背景颜色
    # opt.background_color = np.array([1.0, 1.0, 1.0])  # 背景颜色：白色
    # 增强点云的光泽感
    opt.point_show_normal = True  # 显示法线以增强光照效果
    opt.light_on = True  # 启用光照
    opt.point_color_option = o3d.visualization.PointColorOption.Normal  # 使用法线着色增强光泽
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_lookat(pcd.get_center())

    vis.run()
    vis.destroy_window()

except Exception as e:
    print(f"重建或可视化失败: {e}")
    print("尝试以下操作:")
    print("- 增大depth值（如12-15）")
    print("- 调整scale值（如1.2-1.5）")
    print("- 升级Open3D到最新版本")

# ------------------------- 备用方案：Alpha形状 --------------------
if len(mesh.vertices) == 0:
    print("->尝试Alpha形状重建...")
    alpha = 0.005
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    if len(mesh.vertices) > 0:
        print(f"Alpha={alpha} 重建成功，顶点数:{len(mesh.vertices)}")

        # 设置银白色金属质感
        mesh.paint_uniform_color(silver_color)

        # 使用相同的可视化设置
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1200, height=800)
        vis.add_geometry(mesh)

        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.background_color = np.array([0, 0, 0])  # 黑色背景
        opt.light_on = True
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal

        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_lookat(pcd.get_center())

        vis.run()
        vis.destroy_window()
















