import numpy as np
import open3d as o3d
import time
import math
import os
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def detect_vertical_surfaces(pcd):
    """专门针对长方体池子结构的竖直面检测"""
    print("\n===== 长方体池子结构检测 =====")

    # 获取点云数据
    points = np.asarray(pcd.points)
    print(f"处理点云: {len(points)}个点")

    # 1. 使用PCA确定主方向
    pca = PCA(n_components=3)
    pca.fit(points)
    eigvals = pca.explained_variance_
    eigvecs = pca.components_

    print("PCA主方向:")
    for i in range(3):
        print(f"  方向{i + 1}: {eigvecs[i]}, 方差: {eigvals[i]:.2f}")

    # 2. 确定竖直方向（Z轴）
    # 找到变化最小的方向作为竖直方向
    vertical_idx = np.argmin(eigvals)
    vertical_dir = eigvecs[vertical_idx]
    print(f"检测到竖直方向: {vertical_dir}")

    # 3. 投影点云到水平面
    proj_points = points.copy()
    # 将竖直方向分量置零
    proj_points = proj_points - np.outer(np.dot(proj_points, vertical_dir), vertical_dir)

    # 4. 在投影平面上检测边界
    # 使用DBSCAN聚类检测边界区域
    db = DBSCAN(eps=0.5, min_samples=10).fit(proj_points)
    labels = db.labels_

    # 5. 分析聚类结果
    clusters = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue

        # 获取当前聚类的点索引
        cluster_indices = np.where(labels == label)[0]
        cluster_size = len(cluster_indices)

        if cluster_size < 100:  # 过滤小聚类
            continue

        # 创建聚类点云
        cluster_pcd = pcd.select_by_index(cluster_indices)
        cluster_points = np.asarray(cluster_pcd.points)

        # 计算聚类中心
        centroid = np.mean(cluster_points, axis=0)

        # 计算边界框
        min_bound = np.min(cluster_points, axis=0)
        max_bound = np.max(cluster_points, axis=0)
        bbox_size = max_bound - min_bound

        # 检查是否为竖直面（一个维度远小于其他两个维度）
        size_ratio = bbox_size / np.max(bbox_size)
        if np.min(size_ratio) < 0.1:  # 最薄维度小于最大维度的10%
            # 找到最薄维度的方向
            thin_dim = np.argmin(bbox_size)
            normal = np.zeros(3)
            normal[thin_dim] = 1.0

            clusters.append({
                "indices": cluster_indices,
                "size": cluster_size,
                "centroid": centroid,
                "normal": normal,
                "point_cloud": cluster_pcd,
                "bbox_size": bbox_size
            })
            print(f"检测到潜在竖直面: {cluster_size}点, 尺寸: {bbox_size}")

    # 6. 过滤并排序结果
    vertical_clusters = []
    for cluster in clusters:
        # 计算法线与竖直方向的夹角
        angle = np.arccos(np.clip(np.dot(cluster["normal"], vertical_dir), -1.0, 1.0)) * 180 / np.pi
        if abs(angle) < 30 or abs(angle - 180) < 30:  # 允许30度偏差
            vertical_clusters.append(cluster)
            print(f"确认竖直面: 点数={cluster['size']}, 法线角度={angle:.1f}°")

    # 按尺寸排序（最大面优先）
    vertical_clusters.sort(key=lambda x: np.max(x["bbox_size"]), reverse=True)

    print(f"共检测到 {len(vertical_clusters)} 个竖直面")
    return vertical_clusters


def visualize_pool_structure(pcd, clusters):
    """可视化池子结构检测结果"""
    print("\n准备可视化池子结构...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='池子结构检测', width=1200, height=800)

    # 添加原始点云（灰色）
    base_pcd = pcd
    if len(pcd.points) > 50000:
        base_pcd = pcd.voxel_down_sample(0.05)
    base_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 浅灰色背景
    vis.add_geometry(base_pcd)

    # 添加池底（绿色）
    if clusters:
        # 找到最低的聚类作为池底
        min_z = float('inf')
        bottom_cluster = None
        for cluster in clusters:
            centroid = cluster["centroid"]
            if centroid[2] < min_z:
                min_z = centroid[2]
                bottom_cluster = cluster

        if bottom_cluster:
            bottom_pcd = bottom_cluster["point_cloud"]
            bottom_pcd.paint_uniform_color([0.0, 0.8, 0.0])  # 绿色池底
            vis.add_geometry(bottom_pcd)
            clusters.remove(bottom_cluster)

    # 添加上沿（黄色）
    if clusters:
        # 找到最高的聚类作为上沿
        max_z = float('-inf')
        top_cluster = None
        for cluster in clusters:
            centroid = cluster["centroid"]
            if centroid[2] > max_z:
                max_z = centroid[2]
                top_cluster = cluster

        if top_cluster:
            top_pcd = top_cluster["point_cloud"]
            top_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色上沿
            vis.add_geometry(top_pcd)
            clusters.remove(top_cluster)

    # 添加侧面（不同颜色）
    side_colors = [
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 0.0, 1.0],  # 蓝色
        [1.0, 0.0, 1.0],  # 紫色
        [0.0, 1.0, 1.0]  # 青色
    ]

    for i, cluster in enumerate(clusters[:4]):  # 最多显示4个侧面
        side_pcd = cluster["point_cloud"]
        color_idx = i % len(side_colors)
        side_pcd.paint_uniform_color(side_colors[color_idx])
        vis.add_geometry(side_pcd)

        # 添加法线指示
        normal = cluster["normal"]
        centroid = cluster["centroid"]

        # 创建法线箭头
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.1,
            cone_radius=0.2,
            cylinder_height=1.0,
            cone_height=0.4
        )
        arrow.paint_uniform_color(side_colors[color_idx])

        # 定位箭头
        arrow.translate(centroid)

        # 计算旋转（使箭头指向法线方向）
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.arccos(np.dot(z_axis, normal))

        if np.linalg.norm(rotation_axis) > 1e-6:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            arrow.rotate(R, center=centroid)

        vis.add_geometry(arrow)

    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(coord_frame)

    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    print("开始可视化 (按Q退出)...")
    vis.run()
    vis.destroy_window()


def analyze_pool_structure(pcd):
    """分析池子结构特征"""
    print("\n===== 池子结构分析 =====")
    points = np.asarray(pcd.points)
    n_points = len(points)
    print(f"总点数: {n_points}")

    # 计算边界框
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dimensions = max_coords - min_coords
    print(f"边界框尺寸: X={dimensions[0]:.2f}m, Y={dimensions[1]:.2f}m, Z={dimensions[2]:.2f}m")

    # 计算尺寸比例
    size_ratios = dimensions / np.max(dimensions)
    print(f"尺寸比例: X={size_ratios[0]:.2f}, Y={size_ratios[1]:.2f}, Z={size_ratios[2]:.2f}")

    # 分析点云在高度上的分布
    z_values = points[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    z_range = z_max - z_min

    # 计算高度分布直方图
    hist, bin_edges = np.histogram(z_values, bins=50)
    peak_bins = np.argsort(hist)[-2:]  # 找到两个最高峰

    if len(peak_bins) >= 2:
        bottom_level = bin_edges[peak_bins[0]]
        top_level = bin_edges[peak_bins[1]]
        print(f"检测到池底高度: {bottom_level:.2f}m, 池顶高度: {top_level:.2f}m")
        print(f"池深: {abs(top_level - bottom_level):.2f}m")
    else:
        print("警告: 无法识别池底和池顶")

    # 可视化高度分布
    plt.figure(figsize=(10, 6))
    plt.hist(z_values, bins=50, color='blue', alpha=0.7)
    plt.title('点云高度分布')
    plt.xlabel('高度 (m)')
    plt.ylabel('点数量')
    plt.grid(True)
    plt.savefig("height_distribution.png")
    print("已保存高度分布图: height_distribution.png")

    # 保存点云切片用于检查
    if z_range > 0:
        # 底部切片
        bottom_slice = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[min_coords[0], min_coords[1], z_min],
            max_bound=[max_coords[0], max_coords[1], z_min + z_range * 0.1]
        ))
        o3d.io.write_point_cloud("bottom_slice.ply", bottom_slice)

        # 顶部切片
        top_slice = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[min_coords[0], min_coords[1], z_max - z_range * 0.1],
            max_bound=[max_coords[0], max_coords[1], z_max]
        ))
        o3d.io.write_point_cloud("top_slice.ply", top_slice)

        print(f"已保存底部切片: bottom_slice.ply ({len(bottom_slice.points)}点)")
        print(f"已保存顶部切片: top_slice.ply ({len(top_slice.points)}点)")


def manual_pool_selection(pcd):
    """手动选择池子结构"""
    print("\n===== 手动池子结构选择 =====")
    print("请按以下步骤操作:")
    print("1. 按 'P' 进入点选择模式")
    print("2. 按住Ctrl+鼠标左键选择池底区域")
    print("3. 按Shift+鼠标左键拖动选择池壁区域")
    print("4. 选择完成后按 'Q' 退出")

    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name='手动选择池子结构')

    # 添加点云
    vis.add_geometry(pcd)
    vis.run()  # 用户交互

    # 获取选择结果
    selected_indices = vis.get_picked_points()
    vis.destroy_window()

    if not selected_indices:
        print("未选择任何点")
        return []

    # 创建选择的点云
    selected_pcd = pcd.select_by_index(selected_indices)

    # 尝试拟合平面
    try:
        plane_model, inliers = selected_pcd.segment_plane(
            distance_threshold=0.2,
            ransac_n=5,
            num_iterations=2000
        )

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-6:
            normal /= normal_norm

        # 创建结果
        cluster = {
            "indices": inliers,
            "point_cloud": selected_pcd.select_by_index(inliers),
            "normal": normal
        }

        return [cluster]
    except:
        print("平面拟合失败，返回原始选择")
        cluster = {
            "indices": selected_indices,
            "point_cloud": selected_pcd,
            "normal": np.array([0, 0, 1])
        }
        return [cluster]


def main():
    print("===== 池子结构检测系统 =====")
    print("当前时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))

    # 加载点云
    filename = "pointcloudfile/2025_05_09_07_59_41.ply"
    print(f"加载点云: {filename}")

    if not os.path.exists(filename):
        print(f"错误: 文件不存在 - {filename}")
        return

    try:
        pcd = o3d.io.read_point_cloud(filename)
        print(f"原始点数: {len(pcd.points)}")
    except Exception as e:
        print(f"加载错误: {str(e)}")
        return

    # 分析池子结构特征
    analyze_pool_structure(pcd)

    # 创建备份用于后续处理
    original_pcd = copy.deepcopy(pcd)

    # 自动检测池子结构
    clusters = detect_vertical_surfaces(pcd)

    # 如果没有检测到，启动手动模式
    if not clusters:
        print("\n自动检测未找到池子结构，启动手动选择模式...")
        clusters = manual_pool_selection(original_pcd)

    # 可视化结果
    if clusters:
        print(f"\n共检测到 {len(clusters)} 个池子结构面:")
        for i, cluster in enumerate(clusters):
            normal = cluster.get("normal", [0, 0, 0])
            print(f"  面#{i + 1}: 点数={len(cluster['indices'])}, 法线={normal}")

        visualize_pool_structure(original_pcd, clusters)
    else:
        print("\n未检测到任何池子结构")
        # 可视化原始点云
        o3d.visualization.draw_geometries([original_pcd])

    print("\n处理完成!")


if __name__ == "__main__":
    main()