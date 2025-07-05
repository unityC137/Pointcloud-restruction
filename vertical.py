import numpy as np
import open3d as o3d
import time
import math
import os
import matplotlib.pyplot as plt
import gc  # 垃圾回收


def analyze_large_scale_pcd(pcd):
    """专为大型结构设计的点云分析（内存优化版）"""
    points = np.asarray(pcd.points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    dimensions = max_bound - min_bound

    # 计算点云密度
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    density = len(points) / volume if volume > 0 else 0

    print("\n===== 点云结构分析 =====")
    print(f"尺寸: X={dimensions[0]:.1f}m, Y={dimensions[1]:.1f}m, Z={dimensions[2]:.1f}m")
    print(f"长宽比: {max(dimensions[0] / dimensions[1], dimensions[1] / dimensions[0]):.1f}:1")
    print(f"高宽比: {dimensions[2] / max(dimensions[0], dimensions[1]):.2f}")
    print(f"点密度: {density:.2f} points/m³")

    # 分析高度分布（使用分块处理避免内存问题）
    z_values = points[:, 2]
    hist, bins = np.histogram(z_values, bins=50)
    max_bin = bins[np.argmax(hist)]
    print(f"最高密度高度: {max_bin:.1f}m ({np.max(hist)}点)")

    # 释放内存
    del points, z_values
    gc.collect()

    return min_bound, max_bound, dimensions


def detect_horizontal_surfaces(pcd, min_bound, max_bound, dimensions):
    """检测水平表面（避免聚类算法）"""
    print("\n===== 水平表面检测 =====")

    # 根据尺寸特征选择参数
    xy_size = max(dimensions[0], dimensions[1])
    z_size = dimensions[2]

    # 自动设置参数
    xy_resolution = max(0.5, xy_size * 0.01)  # XY分辨率约为总尺寸的1%
    z_tolerance = max(0.1, z_size * 0.005)  # Z容差约为高度0.5%
    min_region_size = max(500, len(pcd.points) * 0.001)  # 最小区域大小

    print(f"使用参数: XY分辨率={xy_resolution:.2f}m, Z容差={z_tolerance:.2f}m, 最小区域={min_region_size:.0f}点")

    surfaces = []
    height_levels = np.linspace(min_bound[2], max_bound[2], num=20)

    for level in height_levels:
        # 提取高度层附近点
        z_min = level - z_tolerance
        z_max = level + z_tolerance

        # 创建高度层点云
        height_pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-np.inf, -np.inf, z_min],
            max_bound=[np.inf, np.inf, z_max]
        ))

        if len(height_pcd.points) < min_region_size:
            continue

        # 使用Open3D的聚类算法（更高效）
        labels = np.array(height_pcd.cluster_dbscan(eps=xy_resolution * 2, min_points=10, print_progress=False))

        if len(labels) == 0:
            continue

        max_label = labels.max()
        if max_label < 0:  # 没有有效聚类
            continue

        for label in range(0, max_label + 1):
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < min_region_size:
                continue

            cluster_pcd = height_pcd.select_by_index(cluster_indices)
            cluster_points = np.asarray(cluster_pcd.points)

            # 计算聚类属性
            centroid = np.mean(cluster_points, axis=0)
            min_xy = np.min(cluster_points[:, :2], axis=0)
            max_xy = np.max(cluster_points[:, :2], axis=0)
            area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
            avg_height = np.mean(cluster_points[:, 2])

            surfaces.append({
                "type": "水平表面",
                "avg_height": avg_height,
                "area": area,
                "points": cluster_points,
                "xy_range": (min_xy, max_xy)
            })
            print(f"检测到水平表面: 高度={avg_height:.2f}m, 面积={area:.1f}㎡, 点数={len(cluster_points)}")

        # 释放内存
        del height_pcd, labels
        gc.collect()

    return surfaces


def detect_vertical_walls(pcd, surfaces, min_bound, max_bound, dimensions):
    """检测垂直墙壁（避免聚类算法）"""
    print("\n===== 垂直墙壁检测 =====")

    # 根据尺寸特征选择参数
    xy_size = max(dimensions[0], dimensions[1])
    xy_resolution = max(0.5, xy_size * 0.01)  # XY分辨率约为总尺寸的1%
    min_region_size = max(500, len(pcd.points) * 0.001)  # 最小区域大小

    walls = []

    # 检查池底是否存在
    if surfaces:
        # 取最密集的表面作为池底
        main_surface = max(surfaces, key=lambda s: s["area"])
        min_xy, max_xy = main_surface["xy_range"]

        # 池底边界区域
        wall_margin = xy_resolution * 5  # 池壁检测范围

        # 四个池壁方向
        directions = [
            {"name": "西墙", "x_range": (min_xy[0] - wall_margin, min_xy[0] + wall_margin),
             "y_range": (min_xy[1], max_xy[1])},
            {"name": "东墙", "x_range": (max_xy[0] - wall_margin, max_xy[0] + wall_margin),
             "y_range": (min_xy[1], max_xy[1])},
            {"name": "南墙", "x_range": (min_xy[0], max_xy[0]),
             "y_range": (min_xy[1] - wall_margin, min_xy[1] + wall_margin)},
            {"name": "北墙", "x_range": (min_xy[0], max_xy[0]),
             "y_range": (max_xy[1] - wall_margin, max_xy[1] + wall_margin)}
        ]

        for direction in directions:
            # 提取边缘区域点
            edge_pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
                min_bound=[direction["x_range"][0], direction["y_range"][0], min_bound[2]],
                max_bound=[direction["x_range"][1], direction["y_range"][1], max_bound[2]]
            ))

            if len(edge_pcd.points) < min_region_size:
                continue

            # 直接计算高度范围（避免聚类）
            points = np.asarray(edge_pcd.points)
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])
            height = z_max - z_min

            # 检查点分布是否垂直
            if height > xy_resolution * 2:  # 高宽比判定
                walls.append({
                    "type": "垂直表面",
                    "direction": direction["name"],
                    "height": height,
                    "min_z": z_min,
                    "max_z": z_max,
                    "points": points
                })
                print(f"检测到{direction['name']}: 高度={height:.2f}m, 点数={len(points)}")

            # 释放内存
            del edge_pcd
            gc.collect()

    return walls


def visualize_structural_analysis(pcd, surfaces, walls):
    """可视化结构分析结果（优化版）"""
    print("\n准备可视化分析结果...")

    # 创建可视化实体列表
    geometries = []

    # 添加原始点云（半透明）
    pcd_vis = pcd
    if len(pcd.points) > 100000:
        pcd_vis = pcd.voxel_down_sample(0.1)
    pcd_vis.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(pcd_vis)

    # 添加水平表面（绿色）
    for i, surface in enumerate(surfaces):
        # 创建平面表示
        min_xy, max_xy = surface["xy_range"]
        width = max_xy[0] - min_xy[0]
        height_val = max_xy[1] - min_xy[1]

        # 仅当尺寸合理时创建
        if width > 0 and height_val > 0:
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=width,
                height=height_val,
                depth=0.1
            )
            mesh.translate([
                min_xy[0],
                min_xy[1],
                surface["avg_height"] - 0.05  # 稍微低于实际高度
            ])

            # 设置颜色和透明度
            mesh.paint_uniform_color([0.0, 1.0, 0.0])
            mesh.compute_vertex_normals()

            geometries.append(mesh)

    # 添加垂直墙（蓝色）
    for wall in walls:
        # 创建墙的表示
        wall_points = np.asarray(wall["points"])

        # 创建墙壁的线框
        min_pt = np.min(wall_points, axis=0)
        max_pt = np.max(wall_points, axis=0)

        # 创建线框盒子
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_pt, max_pt)
        bbox.color = [0.0, 0.0, 1.0]  # 蓝色

        geometries.append(bbox)

    # 添加比例参考线
    origin = min_bound
    axis_size = max(dimensions) * 0.1
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=origin - [axis_size, axis_size, 0]
    )
    geometries.append(coord_frame)

    # 创建可视化窗口
    o3d.visualization.draw_geometries(geometries, window_name="结构分析结果")

    # 保存结果截图
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for geom in geometries:
        vis.add_geometry(geom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("structural_analysis.jpg")
    vis.destroy_window()

    print("已保存结构分析截图: structural_analysis.jpg")


def export_structural_report(surfaces, walls, dimensions):
    """导出结构分析报告"""
    with open("structure_report.html", "w") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>点云结构分析报告</title>\n")
        f.write("<style>body {font-family: Arial, sans-serif; margin: 20px;}</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>点云结构分析报告</h1>\n")
        f.write(f"<p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        f.write("<h2>结构尺寸</h2>\n")
        f.write(f"<ul><li>长度: {dimensions[0]:.1f}m</li>")
        f.write(f"<li>宽度: {dimensions[1]:.1f}m</li>")
        f.write(f"<li>高度: {dimensions[2]:.1f}m</li></ul>\n")

        f.write("<h2>水平表面</h2>\n")
        for i, surface in enumerate(surfaces):
            f.write(f"<h3>表面 #{i + 1}</h3>\n")
            f.write(f"<ul><li>高度: {surface['avg_height']:.2f}m</li>")
            f.write(f"<li>面积: {surface['area']:.1f}㎡</li>")
            f.write(f"<li>点数: {len(surface['points'])}</li></ul>\n")

        f.write("<h2>垂直墙壁</h2>\n")
        for wall in walls:
            f.write(f"<h3>{wall['direction']}</h3>\n")
            f.write(f"<ul><li>高度: {wall['height']:.1f}m</li>")
            f.write(f"<li>点数: {len(wall['points'])}</li>")
            f.write(f"<li>底部高度: {wall['min_z']:.2f}m</li>")
            f.write(f"<li>顶部高度: {wall['max_z']:.2f}m</li></ul>\n")

        f.write("<h2>结构分析图</h2>\n")
        f.write('\n')

        f.write("</body>\n</html>")

    print("已保存结构分析报告: structure_report.html")


def main():
    print("===== 大型结构分析系统 =====")
    start_time = time.time()
    global min_bound, dimensions  # 用于可视化

    # 加载点云
    filename = "2025_04_27_18_12_33.ply"
    print(f"加载点云: {filename}")

    try:
        pcd = o3d.io.read_point_cloud(filename)
        print(f"原始点数: {len(pcd.points)}")

        # 下采样点云以节省内存
        if len(pcd.points) > 100000:
            print("执行初步下采样...")
            pcd = pcd.voxel_down_sample(1.0)  # 增大下采样体素尺寸
            print(f"下采样后点数: {len(pcd.points)}")
    except Exception as e:
        print(f"加载错误: {str(e)}")
        return

    # 分析点云结构
    min_bound, max_bound, dimensions = analyze_large_scale_pcd(pcd)

    # 处理点云 - 水平表面
    surfaces = detect_horizontal_surfaces(pcd, min_bound, max_bound, dimensions)

    # 处理点云 - 垂直墙壁
    walls = detect_vertical_walls(pcd, surfaces, min_bound, max_bound, dimensions)

    # 可视化结果
    visualize_structural_analysis(pcd, surfaces, walls)

    # 导出报告
    export_structural_report(surfaces, walls, dimensions)

    print(f"处理完成! 总耗时: {time.time() - start_time:.2f}秒")


if __name__ == "__main__":
    main()












