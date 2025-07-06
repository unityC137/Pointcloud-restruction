import numpy as np
import open3d as o3d
import trimesh
import os
import time
import shutil
import argparse


###stl转为ply修复结构

def direct_triangle_points(stl_path, output_path, points_per_triangle=100):
    """
    直接将每个三角形转换为固定数量的点云
    points_per_triangle: 每个三角形生成的固定点数
    """
    print(f"加载STL文件: {stl_path}")
    start_time = time.time()

    # 安全加载模型
    try:
        stl_mesh = trimesh.load(stl_path)
    except Exception as e:
        print(f" 加载失败: {str(e)}")
        return None

    # 模型基本信息
    faces = stl_mesh.faces
    vertices = stl_mesh.vertices
    print(f"三角面数量: {len(faces):,}")
    print(f"顶点数量: {len(vertices):,}")

    # 计算总点数
    total_points = len(faces) * points_per_triangle
    print(f"生成固定点云: {total_points:,}点 ({points_per_triangle}点/三角面)")

    # 使用重心坐标法为每个三角形生成固定数量的点
    all_points = []

    print("🛠 开始三角形点云生成...")

    # 预先生成重心坐标的随机数
    r1 = np.random.random((points_per_triangle, len(faces)))
    r2 = np.random.random((points_per_triangle, len(faces)))
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    # 分批处理
    batch_size = 50000
    num_batches = (len(faces) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(faces))
        batch_faces = faces[start_idx:end_idx]

        batch_points = []

        for i, face in enumerate(batch_faces):
            # 获取三角形顶点
            v0, v1, v2 = vertices[face]

            # 使用预计算的坐标计算每个点
            for j in range(points_per_triangle):
                # 计算当前点的重心坐标
                point = (u[j, start_idx + i] * v0 +
                         v[j, start_idx + i] * v1 +
                         w[j, start_idx + i] * v2)
                batch_points.append(point)

        # 添加当前批次的所有点
        all_points.extend(batch_points)
        print(f"✓ 已完成批次 {batch_idx + 1}/{num_batches}, 点数: {len(all_points):,}")

    # 转换为numpy数组
    point_array = np.array(all_points)

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)

    # 保存完整点云
    print(f"保存点云 ({len(point_array):,}点) 到: {output_path}")
    try:
        o3d.io.write_point_cloud(output_path, pcd)
    except Exception as e:
        print(f"保存失败: {e}")
        return None

    # 性能统计
    end_time = time.time()
    process_time = end_time - start_time
    print(f"⏱️ 处理时间: {process_time:.1f}秒 | 点生成速率: {len(point_array) / process_time:,.0f}点/秒")

    # 创建预览
    if len(point_array) > 100000:
        preview_path = output_path.replace(os.path.splitext(output_path)[1], "_PREVIEW.ply")
        preview_points = point_array[:100000]
        preview_pcd = o3d.geometry.PointCloud()
        preview_pcd.points = o3d.utility.Vector3dVector(preview_points)
        o3d.io.write_point_cloud(preview_path, preview_pcd)
        print(f"已创建预览点云: {preview_path}")

    return pcd


def visualize_point_cloud(pcd, max_points=100000):
    """安全可视化点云"""
    if pcd is None:
        print("⚠️ 无可视化点云数据")
        return

    points = np.asarray(pcd.points)
    num_points = len(points)

    if num_points == 0:
        print("⚠️ 点云为空")
        return

    # 如果点云太大，取子集可视化
    if num_points > max_points:
        print(f"🔍 点云规模过大 ({num_points:,}点)，仅显示前 {max_points} 个点")
        display_points = points[:max_points]
    else:
        display_points = points

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(display_points)

    try:
        o3d.visualization.draw_geometries([o3d_pcd])
    except Exception as e:
        print(f"可视化失败: {e}")
        print("建议使用CloudCompare或专业点云软件查看完整点云")


if __name__ == "__main__":
    # 记录整个程序开始时间
    overall_start_time = time.time()

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='STL文件转换为PLY点云文件')

    # 添加STL文件路径参数
    parser.add_argument('-i', '--input', required=True, help='输入STL文件路径')

    # 添加输出文件路径参数
    parser.add_argument('-o', '--output', required=True, help='输出文件路径')

    # 添加每个三角形生成点数的可选参数
    parser.add_argument('-p', '--points', type=int, default=50,
                        help='每个三角形生成的点数 (默认: 50)')
    parser.add_argument('-v', '--visual', action='store_true', help='显示open3d的显示界面')
    # 解析命令行参数
    args = parser.parse_args()

    # 获取命令行参数
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    POINTS_PER_TRIANGLE = args.points

    print("=" * 60)
    print(f"开始处理文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"每个三角形生成点数: {POINTS_PER_TRIANGLE}")
    print("=" * 60)

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"错误: 输入文件 '{INPUT_PATH}' 不存在")
        exit(1)

    # 检查输出文件格式
    valid_formats = ['.ply', '.pcd', '.xyz', '.xyzrgb', '.xyzn', '.pts']
    file_ext = os.path.splitext(OUTPUT_PATH)[1].lower()

    if file_ext not in valid_formats:
        print(f"警告: 不常见的输出格式 '{file_ext}'。支持的格式有: {', '.join(valid_formats)}")

    # 直接为每个三角形生成固定数量的点
    result_pcd = direct_triangle_points(INPUT_PATH, OUTPUT_PATH, POINTS_PER_TRIANGLE)

    # 如果转换失败，尝试仅提取顶点
    if result_pcd is None or len(result_pcd.points) == 0:
        print("\n 尝试仅提取顶点...")
        try:
            stl_mesh = trimesh.load(INPUT_PATH)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(stl_mesh.vertices)
            o3d.io.write_point_cloud(OUTPUT_PATH, pcd)
            print(f"保存顶点云 ({len(stl_mesh.vertices):,}点) 到: {OUTPUT_PATH}")
            result_pcd = pcd
        except Exception as e:
            print(f"顶点提取失败: {e}")
            result_pcd = None

    # 可视化
    if args.visual and result_pcd:
        visualize_point_cloud(result_pcd)

    # 记录整个程序结束时间
    overall_end_time = time.time()
    overall_process_time = overall_end_time - overall_start_time
    print(f"⏱️ 整个程序执行时间: {overall_process_time:.1f} 秒")

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)