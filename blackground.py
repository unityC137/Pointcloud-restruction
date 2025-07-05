import open3d as o3d
import numpy as np
import time
import concurrent.futures
import multiprocessing as mp
import tqdm

def timeit(func):
    """性能计时装饰器"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 耗时: {end_time - start_time:.2f} 秒")
        return result

    return wrapper


def process_cell(points_array, indices, min_bound, max_bound, nb_neighbors, std_ratio):
    """处理单个网格单元的函数"""
    points = points_array[indices]
    cell_pcd = o3d.geometry.PointCloud()
    cell_pcd.points = o3d.utility.Vector3dVector(points)
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array(min_bound), np.array(max_bound))
    cell_pcd = cell_pcd.crop(bbox)

    if len(cell_pcd.points) > 0:
        filtered_cell, _ = cell_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return np.asarray(filtered_cell.points)
    return None


def process_cell_wrapper(args):
    """处理单元的包装函数，避免使用lambda"""
    return process_cell(*args)


@timeit
def load_and_preprocess_point_cloud(file_path):
    """加载并预处理点云，支持并行滤波"""
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError("点云为空，请检查文件路径")

    print(f"原始点云点数: {len(pcd.points):,}")
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"点云范围: {bbox}")

    print("->正在进行点云预处理...")
    if len(pcd.points) > 1e6:
        print("->点云过大，使用并行滤波...")
        points = np.asarray(pcd.points)
        filtered_points = parallel_statistical_filter(points, bbox, nb_neighbors=10, std_ratio=2.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
    else:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.5)

    print(f"滤波后点数: {len(pcd.points):,}")

    if len(pcd.points) > 5e6:
        voxel_size = max(0.001, bbox.get_extent().max() / 1000)
        print(f"->点云过大，使用体素降采样 (voxel_size={voxel_size:.6f})")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"降采样后点数: {len(pcd.points):,}")

    return pcd


def parallel_statistical_filter(points, bbox, nb_neighbors=10, std_ratio=2.5, num_workers=4):
    """并行统计滤波，加速大点云处理"""
    extent = bbox.get_extent()
    center = bbox.get_center()
    grid_size = 2
    grid_cells = []

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                min_bound = [
                    center[0] - extent[0] / 2 + i * extent[0] / grid_size,
                    center[1] - extent[1] / 2 + j * extent[1] / grid_size,
                    center[2] - extent[2] / 2 + k * extent[2] / grid_size
                ]
                max_bound = [
                    center[0] - extent[0] / 2 + (i + 1) * extent[0] / grid_size,
                    center[1] - extent[1] / 2 + (j + 1) * extent[1] / grid_size,
                    center[2] - extent[2] / 2 + (k + 1) * extent[2] / grid_size
                ]
                grid_cells.append((min_bound, max_bound))

    grid_indices = []
    for min_bound, max_bound in grid_cells:
        mask = np.logical_and(
            np.all(points >= np.array(min_bound), axis=1),
            np.all(points <= np.array(max_bound), axis=1)
        )
        indices = np.where(mask)[0]
        grid_indices.append(indices)

    args_list = []
    for i, (min_bound, max_bound) in enumerate(grid_cells):
        indices = grid_indices[i]
        if len(indices) > 0:
            args_list.append((points, indices, min_bound, max_bound, nb_neighbors, std_ratio))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_cell_wrapper, args_list),
                            total=len(args_list), desc="并行滤波"))

    filtered_points = []
    for result in results:
        if result is not None:
            filtered_points.append(result)

    return np.vstack(filtered_points) if filtered_points else np.empty((0, 3))


@timeit
def estimate_normals(pcd):
    """估计法向量，优化参数和算法"""
    print("->正在估计法向量...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    avg_distance = compute_average_distance(pcd, pcd_tree, sample_size=1000)
    radius = max(0.005, avg_distance * 5)
    print(f"->使用自适应法向量半径: {radius:.6f}")

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=15)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def compute_average_distance(pcd, pcd_tree, sample_size=1000):
    """计算点云平均距离，采样以提高效率"""
    points = np.asarray(pcd.points)
    indices = np.random.choice(len(points), min(sample_size, len(points)), replace=False)
    distances = []
    for i in indices:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)
        if k > 1:
            avg_dist = np.mean([np.linalg.norm(points[i] - points[j]) for j in idx[1:]])
            distances.append(avg_dist)
    return np.mean(distances) if distances else 0.001


@timeit
def reconstruct_surface(pcd):
    """使用多种方法进行表面重建，选择最优方案"""
    print("->开始三维重建...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    avg_distance = compute_average_distance(pcd, pcd_tree)
    methods = ["alpha_shape", "poisson", "bpa"]
    mesh = None

    for method in methods:
        try:
            print(f"\n->尝试{method}重建...")
            if method == "alpha_shape":
                alpha_values = [avg_distance * 10, avg_distance * 20, avg_distance * 50]
                print(f"尝试Alpha值: {[round(a, 6) for a in alpha_values]}")
                for alpha in alpha_values:
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                    if len(mesh.vertices) > 0:
                        print(f"Alpha={alpha:.6f} 重建成功，顶点数: {len(mesh.vertices):,}")
                        break
            elif method == "poisson":
                depth = min(10, int(np.log2(len(pcd.points) / 1e4)) + 8)
                print(f"->使用泊松重建 (depth={depth})")
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, scale=1.1, width=0
                )
            elif method == "bpa":
                radius = 3 * avg_distance
                print(f"->使用BPA算法 (radius={radius:.6f})")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector([radius, radius * 2])
                )

            if mesh is not None and len(mesh.vertices) > 0:
                print(f"{method}重建成功，顶点数: {len(mesh.vertices):,}")
                break
        except Exception as e:
            print(f"{method}重建失败: {e}")

    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError("所有重建方法均失败，请检查点云质量")
    return mesh


import open3d as o3d
import numpy as np
import time
import concurrent.futures

import multiprocessing as mp

# 检查Open3D版本
O3D_VERSION = o3d.__version__.split('.')
HAS_CURVATURE = int(O3D_VERSION[0]) >= 0 and (int(O3D_VERSION[1]) >= 10)


def timeit(func):
    """性能计时装饰器"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 耗时: {end_time - start_time:.2f} 秒")
        return result

    return wrapper


@timeit
def load_and_preprocess_point_cloud(file_path):
    """加载并预处理点云"""
    # 正确调用：传入文件路径
    pcd = o3d.io.read_point_cloud(file_path)  # 这里必须是文件路径字符串
    if pcd.is_empty():
        raise ValueError("点云文件为空，请检查路径")

    print(f"原始点云点数: {len(pcd.points):,}")
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"点云范围: {bbox}")

    print("->正在进行点云预处理...")
    if len(pcd.points) > 1e6:
        print("->点云过大，使用并行滤波...")
        points = np.asarray(pcd.points)
        filtered_points = parallel_statistical_filter(points, bbox)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
    else:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.5)

    print(f"滤波后点数: {len(pcd.points):,}")

    if len(pcd.points) > 5e6:
        voxel_size = max(0.001, bbox.get_extent().max() / 1000)
        print(f"->点云过大，使用体素降采样 (voxel_size={voxel_size:.6f})")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"降采样后点数: {len(pcd.points):,}")

    return pcd


def parallel_statistical_filter(points, bbox, nb_neighbors=10, std_ratio=2.5, num_workers=4):
    """并行统计滤波"""
    extent = bbox.get_extent()
    center = bbox.get_center()
    grid_size = 2
    grid_cells = []

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                min_bound = [
                    center[0] - extent[0] / 2 + i * extent[0] / grid_size,
                    center[1] - extent[1] / 2 + j * extent[1] / grid_size,
                    center[2] - extent[2] / 2 + k * extent[2] / grid_size
                ]
                max_bound = [
                    center[0] - extent[0] / 2 + (i + 1) * extent[0] / grid_size,
                    center[1] - extent[1] / 2 + (j + 1) * extent[1] / grid_size,
                    center[2] - extent[2] / 2 + (k + 1) * extent[2] / grid_size
                ]
                grid_cells.append((min_bound, max_bound))

    grid_indices = []
    for min_bound, max_bound in grid_cells:
        mask = np.logical_and(
            np.all(points >= np.array(min_bound), axis=1),
            np.all(points <= np.array(max_bound), axis=1)
        )
        indices = np.where(mask)[0]
        grid_indices.append(indices)

    args_list = []
    for i, (min_bound, max_bound) in enumerate(grid_cells):
        indices = grid_indices[i]
        if len(indices) > 0:
            args_list.append((points, indices, min_bound, max_bound, nb_neighbors, std_ratio))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_cell_wrapper, args_list),
                            total=len(args_list), desc="并行滤波"))

    filtered_points = []
    for result in results:
        if result is not None:
            filtered_points.append(result)

    return np.vstack(filtered_points) if filtered_points else np.empty((0, 3))


def process_cell(points_array, indices, min_bound, max_bound, nb_neighbors, std_ratio):
    """处理单个网格单元"""
    points = points_array[indices]
    cell_pcd = o3d.geometry.PointCloud()
    cell_pcd.points = o3d.utility.Vector3dVector(points)
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array(min_bound), np.array(max_bound))
    cell_pcd = cell_pcd.crop(bbox)

    if len(cell_pcd.points) > 0:
        filtered_cell, _ = cell_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return np.asarray(filtered_cell.points)
    return None


def process_cell_wrapper(args):
    return process_cell(*args)


@timeit
def estimate_normals(pcd):
    """估计法向量"""
    print("->正在估计法向量...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    avg_distance = compute_average_distance(pcd, pcd_tree, sample_size=1000)
    radius = max(0.005, avg_distance * 5)
    print(f"->使用自适应法向量半径: {radius:.6f}")

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=15)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def compute_average_distance(pcd, pcd_tree, sample_size=1000):
    """计算平均距离"""
    points = np.asarray(pcd.points)
    indices = np.random.choice(len(points), min(sample_size, len(points)), replace=False)
    distances = []
    for i in indices:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)
        if k > 1:
            avg_dist = np.mean([np.linalg.norm(points[i] - points[j]) for j in idx[1:]])
            distances.append(avg_dist)
    return np.mean(distances) if distances else 0.001


@timeit
def reconstruct_surface(pcd):
    """表面重建"""
    print("->开始三维重建...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    avg_distance = compute_average_distance(pcd, pcd_tree)
    methods = ["alpha_shape", "poisson", "bpa"]
    mesh = None

    for method in methods:
        try:
            print(f"\n->尝试{method}重建...")
            if method == "alpha_shape":
                alpha_values = [avg_distance * 10, avg_distance * 20, avg_distance * 50]
                for alpha in alpha_values:
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                    if len(mesh.vertices) > 0:
                        print(f"Alpha={alpha:.6f} 重建成功，顶点数: {len(mesh.vertices):,}")
                        break
            elif method == "poisson":
                depth = min(10, int(np.log2(len(pcd.points) / 1e4)) + 8)
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=depth, scale=1.1, width=0
                )
            elif method == "bpa":
                radius = 3 * avg_distance
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector([radius, radius * 2])
                )

            if mesh is not None and len(mesh.vertices) > 0:
                print(f"{method}重建成功，顶点数: {len(mesh.vertices):,}")
                break
        except Exception as e:
            print(f"{method}重建失败: {e}")

    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError("所有重建方法均失败，请检查点云质量")
    return mesh


@timeit
def optimize_mesh(mesh):
    """优化网格"""
    print("->优化网格...")
    vertex_count = len(mesh.vertices)
    if vertex_count > 1e6:
        target_count = min(5e5, vertex_count // 2)
        print(f"->简化网格 (从{vertex_count:,}到{int(target_count):,}顶点)")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_count // 2))

    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()
    return mesh


def enhance_edge_details(pcd, edge_threshold=0.1):
    """增强边缘细节（兼容新旧版本）"""
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

    if HAS_CURVATURE:
        pcd.estimate_curvature(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        curvatures = np.asarray(pcd.curvature)
        edge_mask = curvatures > edge_threshold
    else:
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        normals = np.asarray(pcd.normals)
        edge_mask = np.zeros(len(pcd.points), dtype=bool)
        for i in range(len(pcd.points)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn=5)
            if k >= 2:
                main_normal = normals[i]
                neighbor_normals = normals[idx[1:]]
                angles = np.arccos(np.clip(np.dot(neighbor_normals, main_normal), -1, 1))
                if np.max(np.degrees(angles)) > 30:
                    edge_mask[i] = True

    edge_pcd = pcd.select_by_index(np.where(edge_mask)[0])
    non_edge_pcd = pcd.select_by_index(np.where(~edge_mask)[0])
    edge_pcd.paint_uniform_color([1, 0, 0])
    non_edge_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    return edge_pcd + non_edge_pcd


def apply_rust_texture(mesh, rust_ratio=0.7, rust_color=[0.6, 0.3, 0.2], metal_color=[0.5, 0.5, 0.5]):
    """应用生锈纹理"""
    vertex_colors = np.array([
        rust_color if np.random.rand() < rust_ratio else metal_color
        for _ in range(len(mesh.vertices))
    ])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def enhance_light_for_rust(vis):
    """调整光照"""
    opt = vis.get_render_option()
    ctr = vis.get_view_control()
    opt.ambient_light = np.array([0.3, 0.2, 0.1])

    if hasattr(vis, 'get_light_profile'):
        light = vis.get_light_profile(0)
        light.set_position([-10000, 10000, 10000])
        light.set_intensity(25000)
        light.set_color([1, 0.9, 0.8])
        vis.set_light_profile(light, 0)

    ctr.set_zoom(0.7)
    ctr.set_front([-0.6, -0.3, 1])
    ctr.set_lookat(vis.get_geometry()[0].get_center())


def visualize_results(pcd, mesh):
    """可视化结果"""
    try:
        enhanced_pcd = enhance_edge_details(pcd, edge_threshold=0.08)
        rust_mesh = apply_rust_texture(mesh.copy())

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1200, height=800)
        vis.add_geometry(enhanced_pcd)
        vis.add_geometry(rust_mesh)

        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.mesh_show_back_face = True
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.mesh_color_option = o3d.visualization.MeshColorOption.VertexColor if HAS_CURVATURE else o3d.visualization.MeshColorOption.Color

        enhance_light_for_rust(vis)
        print("-> 显示结果（红色为边缘，按 Q 退出）")
        vis.run()
        vis.destroy_window()

    except Exception as e:
        print(f"可视化错误: {e}")
        original_vis(pcd, mesh)


def original_vis(pcd, mesh):
    """原始可视化"""
    highlight_color = [0.9, 0.9, 0.9]
    pcd.paint_uniform_color(highlight_color)
    mesh.paint_uniform_color(highlight_color)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0, 0, 0])
    opt.light_on = True

    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    ctr.set_lookat(pcd.get_center())
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')

    try:
        file_path = "output_repaired.ply"  # 确保这是正确的文件路径
        pcd = load_and_preprocess_point_cloud(file_path)  # 传入文件路径
        pcd = estimate_normals(pcd)
        mesh = reconstruct_surface(pcd)
        mesh = optimize_mesh(mesh)
        visualize_results(pcd, mesh)
        print("\n重建完成!")

    except Exception as e:
        print(f"错误: {e}")

