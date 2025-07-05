import numpy as np
import open3d as o3d
import concurrent.futures
from scipy.spatial import Delaunay
from typing import Optional
import argparse
import time
from joblib import Parallel, delayed


# # 尝试导入 trimesh（用于高级网格修复）
# try:
#     import trimesh
#     from trimesh import repair
#
#     TRIMESH_AVAILABLE = True
# except ImportError:
#     TRIMESH_AVAILABLE = False


class MeshOptimizer:
    def __init__(self, input_file: str, output_file: Optional[str] = None, save_as_pcd: bool = False):
        self.input_file = input_file
        self.output_file = output_file
        self.save_as_pcd = save_as_pcd
        self.pcd = None
        self.mesh = None
        self.num_threads = 8  # 默认线程数

        # 检查 Open3D 版本兼容性
        self.supports_copy = hasattr(o3d.geometry.PointCloud, 'copy')

    def load_data(self) -> bool:
        """加载点云或网格数据"""
        try:
            print(f"加载文件: {self.input_file}")
            start_time = time.time()

            if self.input_file.lower().endswith(('.ply', '.pcd')):
                self.pcd = o3d.io.read_point_cloud(self.input_file)
                if not self.pcd.has_points():
                    print(f"错误: 文件 {self.input_file} 不包含点数据")
                    return False

                print(f"成功加载点云，点数: {len(self.pcd.points)}")

                # 检查点云数据是否包含无效值
                points = np.asarray(self.pcd.points)
                if np.isnan(points).any() or np.isinf(points).any():
                    print("警告：点云数据包含NaN或Inf值，进行清理...")
                    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
                    valid_points = points[valid_mask]
                    print(f"移除了 {len(points) - len(valid_points)} 个无效点")

                    if len(valid_points) == 0:
                        print("错误：清理后没有剩余有效点")
                        return False

                    self.pcd.points = o3d.utility.Vector3dVector(valid_points)
                    print(f"清理后的点云点数: {len(self.pcd.points)}")

                print(f"加载耗时: {time.time() - start_time:.2f}秒")
                return True

            elif self.input_file.lower().endswith(('.obj', '.stl', '.off', '.gltf', '.glb')):
                # 网格加载部分保持不变...
                pass

            else:
                print(f"错误: 不支持的文件格式: {self.input_file}")
                return False

        except Exception as e:
            print(f"加载文件时出错: {e}")
            return False

    def adaptive_downsample(self, pcd, target_reduction=0.9, voxel_size_start=0.01, max_iter=5):
        """
        自适应降采样（体素+随机降采样组合）

        参数:
            pcd: 输入点云
            target_reduction: 目标点数减少比例（0.9表示保留10%）
            voxel_size_start: 初始体素大小
            max_iter: 最大尝试次数
        """
        # 检查当前Open3D版本是否支持点云复制
        if not hasattr(pcd, 'copy') or not callable(pcd.copy):
            print("警告：当前Open3D版本不支持点云复制，使用替代方法")
            # 创建一个新的点云对象并复制数据
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
            if pcd.has_normals():
                new_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
            pcd = new_pcd
        else:
            pcd = pcd.copy()

        original_points = len(pcd.points)
        current_reduction = 0.0

        # 计算点云的边界框大小
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_extent()
        min_bbox_dim = min(bbox_size)

        # 设置合理的体素大小下限
        min_voxel_size = min_bbox_dim / 100.0  # 至少将场景分成100个体素

        # 阶段1：体素降采样（逐步调整体素大小）
        voxel_size = voxel_size_start
        for i in range(max_iter):
            if len(pcd.points) <= original_points * (1 - target_reduction):
                break

            # 降采样
            downsampled = pcd.voxel_down_sample(voxel_size)
            print(f"尝试体素大小 {voxel_size:.6f}, 降采样后点数: {len(downsampled.points)}")

            # 检查降采样后的点数是否合理
            if len(downsampled.points) < 10:  # 至少保留10个点
                print(f"警告：体素大小 {voxel_size:.6f} 导致降采样后点数过少 ({len(downsampled.points)})")
                if i == 0:  # 第一次尝试就失败，使用更大的体素
                    voxel_size = max(voxel_size * 2, min_voxel_size)
                else:
                    # 否则使用上一次的结果
                    break
            else:
                pcd = downsampled
                current_reduction = 1 - len(pcd.points) / original_points

            # 调整体素大小
            voxel_size *= 0.5  # 每次尝试减半体素大小
            if voxel_size < min_voxel_size:  # 防止体素过小
                print(f"警告：体素大小已达到最小值 {min_voxel_size:.6f}")
                voxel_size = min_voxel_size

        # 阶段2：随机降采样（确保达到目标点数）
        if current_reduction < target_reduction:
            target_points = int(original_points * (1 - target_reduction))
            if target_points > 0 and len(pcd.points) > target_points:
                # 计算每k个点保留一个
                k = len(pcd.points) // target_points + 1
                pcd = pcd.uniform_down_sample(every_k_points=k)
                current_reduction = 1 - len(pcd.points) / original_points

        print(
            f"降采样完成，点数从 {original_points} 减少到 {len(pcd.points)} ({current_reduction * 100:.2f}% reduction)")

        # 最终检查
        if len(pcd.points) < 10:
            print(f"警告：降采样后点数过少 ({len(pcd.points)})，可能影响后续处理")

        return pcd

    def preprocess_point_cloud(self, voxel_size: float = 0.005, target_reduction=0.9, voxel_size_start=0.01,
                               max_iter=5) -> None:
        """预处理点云数据"""
        if self.pcd is None:
            print("警告：点云数据为空，无法进行预处理")
            return

        print(f"预处理点云 (体素大小: {voxel_size})")
        print(f"预处理前点数: {len(self.pcd.points)}")
        start_time = time.time()

        try:
            original_points = len(self.pcd.points)

            # 计算点云的边界框大小
            bbox = self.pcd.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            min_bbox_dim = min(bbox_size)
            print(f"点云边界框尺寸: {bbox_size}")

            # 检查边界框是否有效
            if np.isnan(bbox_size).any() or np.isinf(bbox_size).any():
                print("警告：边界框计算失败，尝试使用替代方法...")
                points = np.asarray(self.pcd.points)
                if len(points) > 0:
                    min_coords = np.min(points, axis=0)
                    max_coords = np.max(points, axis=0)
                    bbox_size = max_coords - min_coords
                    min_bbox_dim = min(bbox_size)
                    print(f"替代方法计算的边界框尺寸: {bbox_size}")
                else:
                    print("错误：点云数据为空，无法继续处理")
                    return

            # 增加点云过滤，移除离群点
            if len(self.pcd.points) > 1000:  # 确保点云足够大再进行过滤
                print("移除离群点...")

                # 尝试不同的离群点过滤参数
                filtered_pcd, ind = self.pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=3.0)
                filtered_points = len(filtered_pcd.points)

                if filtered_points < original_points * 0.1:  # 如果过滤掉了超过90%的点
                    print(f"警告：离群点过滤太严格，保留了 {filtered_points} 个点，尝试更宽松的参数")
                    filtered_pcd, ind = self.pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=5.0)
                    filtered_points = len(filtered_pcd.points)

                    if filtered_points < original_points * 0.1:  # 仍然过滤太多
                        print(f"警告：宽松的离群点过滤仍然保留了太少的点 ({filtered_points})，使用原始点云")
                        filtered_pcd = self.pcd
                    else:
                        print(f"使用宽松的过滤参数，保留了 {filtered_points} 个点")
                else:
                    print(f"移除了 {original_points - filtered_points} 个离群点")

                self.pcd = filtered_pcd
                print(f"离群点过滤后点数: {len(self.pcd.points)}")

            # ADAPTIVE DOWNSAMPLING
            if len(self.pcd.points) > 0:  # 确保有点可以降采样
                # 计算合理的体素大小
                estimated_voxel_size = max(voxel_size_start, min_bbox_dim / 1000.0)
                print(f"估计的合理体素大小: {estimated_voxel_size}")

                self.pcd = self.adaptive_downsample(self.pcd,
                                                    target_reduction=target_reduction,
                                                    voxel_size_start=estimated_voxel_size,
                                                    max_iter=max_iter)
            else:
                print("错误：离群点过滤后没有剩余点，无法进行降采样")
                return

            # 最终检查
            if len(self.pcd.points) < 1000:  # 提高阈值，确保有足够的点
                print(f"错误：降采样后点云点数太少 ({len(self.pcd.points)})，无法继续处理")
                return

            # 估计法线
            if not self.pcd.has_normals():
                print("估计点云法线...")

                # 根据点云密度调整法线估计参数
                if len(self.pcd.points) < 10000:
                    # 稀疏点云使用更大的搜索半径
                    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
                else:
                    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)

                self.pcd.estimate_normals(search_param=search_param)

                # 法线定向
                try:
                    print("法线定向...")
                    self.pcd.orient_normals_consistent_tangent_plane(50)
                except RuntimeError as e:
                    print(f"法线定向失败: {e}")
                    print("使用替代法线定向方法...")

                    # 替代方法：使用PCA分析来定向法线
                    points = np.asarray(self.pcd.points)
                    normals = np.asarray(self.pcd.normals)

                    # 检查法线是否包含无效值
                    if np.isnan(normals).any() or np.isinf(normals).any():
                        print("警告：法线包含无效值，重新估计...")
                        self.pcd.estimate_normals(search_param=search_param)
                        normals = np.asarray(self.pcd.normals)

                    # 计算每个点的主方向
                    pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

                    for i in range(len(points)):
                        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], min(50, len(points) - 1))
                        if k < 3:  # 确保有足够的点进行PCA
                            continue

                        neighbors = points[idx, :]
                        cov = np.cov(neighbors.T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        # 假设法向量沿着最小特征值对应的特征向量
                        normal = eigenvectors[:, 0]

                        # 确保法线方向一致（这里假设朝上为正方向）
                        if np.dot(normal, [0, 0, 1]) < 0:
                            normal = -normal

                        normals[i] = normal

                    self.pcd.normals = o3d.utility.Vector3dVector(normals)
                    print("已使用PCA方法进行法线定向")

            print(f"点云预处理完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            print(f"点云预处理出错: {e}")
            print("**********************")

    def reconstruct_delaunay_vertical(self,
                                      density_threshold: float = 0.5,
                                      vertical_threshold: float = 30.0,
                                      knn: int = 30) -> bool:
        """
        在点云密集且竖直的区域使用Delaunay三角剖分
        """
        if self.pcd is None or not self.pcd.has_points():
            return False

        print(f"应用Delaunay三角剖分（仅处理密集且竖直的区域），使用{self.num_threads}个线程...")
        start_time = time.time()

        # 确保点云有法线
        if not self.pcd.has_normals():
            print("估计点云法线...")
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=knn)
            )
            # 法线定向
            try:
                self.pcd.orient_normals_consistent_tangent_plane(knn)
            except:
                print("法线定向失败，使用替代方法")
                # 替代方法：假设Z轴朝上
                normals = np.asarray(self.pcd.normals)
                # 计算法线与Z轴的点积
                dot_products = np.dot(normals, np.array([0, 0, 1]))
                # 如果法线与Z轴夹角大于90度，则翻转法线
                flip_mask = dot_products < 0
                normals[flip_mask] = -normals[flip_mask]
                self.pcd.normals = o3d.utility.Vector3dVector(normals)

        points = np.asarray(self.pcd.points)
        normals = np.asarray(self.pcd.normals)

        # 检查点云数据是否包含无效值
        if np.isnan(points).any() or np.isinf(points).any():
            print("警告：点云数据包含NaN或Inf值，进行清理...")
            valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) < 3:
                print("错误：清理后有效点不足，无法进行三角剖分")
                return False

            points = points[valid_indices]
            normals = normals[valid_indices]
            print(f"清理后的有效点数: {len(points)}")

        # 1. 多线程计算点云密度
        print(f"计算点云密度（使用{self.num_threads}个线程）...")
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        densities = np.zeros(len(points))
        batch_size = max(1, len(points) // self.num_threads)
        batches = [range(i, min(i + batch_size, len(points))) for i in range(0, len(points), batch_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_batch = {executor.submit(self._compute_density_batch, batch, points, pcd_tree, knn): batch for
                               batch in batches}
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_indices, batch_densities = future.result()
                densities[batch_indices] = batch_densities

        # 归一化密度
        if np.max(densities) > 0:
            densities = densities / np.max(densities)
        else:
            print("警告：所有点的密度均为0，使用均匀密度...")
            densities = np.ones_like(densities) * 0.5  # 使用0.5作为默认密度

        # 2. 识别竖直区域
        vertical_direction = np.array([0, 0, 1])
        normal_angles = np.arccos(np.abs(np.dot(normals, vertical_direction))) * 180 / np.pi

        # 可视化密度分布和法线角度分布
        print(f"密度分布: min={np.min(densities):.4f}, max={np.max(densities):.4f}, mean={np.mean(densities):.4f}")
        print(
            f"法线角度分布: min={np.min(normal_angles):.2f}, max={np.max(normal_angles):.2f}, mean={np.mean(normal_angles):.2f}")

        # 调整密度阈值和竖直阈值
        dense_mask = densities > density_threshold
        vertical_mask = normal_angles < vertical_threshold
        selected_mask = np.logical_and(dense_mask, vertical_mask)
        selected_indices = np.where(selected_mask)[0]

        print(f"找到 {len(selected_indices)} 个点符合密集且竖直的条件")

        # 如果符合条件的点太少，尝试放宽条件
        if len(selected_indices) < 1000:  # 提高阈值，确保有足够的点
            print("警告：符合条件的点太少，尝试放宽条件...")

            # 尝试降低密度阈值和增加竖直阈值
            relaxed_dense_mask = densities > (density_threshold * 0.5)
            relaxed_vertical_mask = normal_angles < (vertical_threshold * 1.5)
            relaxed_selected_mask = np.logical_and(relaxed_dense_mask, relaxed_vertical_mask)
            relaxed_selected_indices = np.where(relaxed_selected_mask)[0]

            print(f"放宽条件后找到 {len(relaxed_selected_indices)} 个点")

            if len(relaxed_selected_indices) >= 1000:  # 确保有足够的点
                print("使用放宽条件后的点进行三角剖分")
                selected_indices = relaxed_selected_indices
            else:
                # 如果还是不够，尝试只使用竖直条件
                vertical_only_indices = np.where(vertical_mask)[0]
                print(f"仅使用竖直条件找到 {len(vertical_only_indices)} 个点")

                if len(vertical_only_indices) >= 1000:
                    print("使用仅竖直条件的点进行三角剖分")
                    selected_indices = vertical_only_indices
                else:
                    print("错误：无法找到足够的点进行Delaunay三角剖分")
                    self.mesh = None
                    return False

        selected_points = points[selected_indices]

        # 3. 执行Delaunay三角剖分（增加异常处理）
        try:
            # 检查点云分布范围
            point_range = np.ptp(selected_points, axis=0)
            min_range = np.min(point_range)
            print(f"点云分布范围: {point_range}")

            # 检查点云分布是否太集中
            if min_range < 1e-6:
                print("警告：点云分布范围过小，添加抖动...")
                jitter = 1e-6 * np.random.randn(*selected_points.shape)
                selected_points += jitter

                # 重新计算分布范围
                point_range = np.ptp(selected_points, axis=0)
                min_range = np.min(point_range)
                print(f"添加抖动后的点云分布范围: {point_range}")

                if min_range < 1e-6:
                    print("错误：添加抖动后点云分布仍然过小，无法进行Delaunay三角剖分")
                    self.mesh = None
                    return False

            # 投影到XY平面进行2D Delaunay三角剖分
            projected_points = selected_points[:, :2]

            # 添加微小抖动以避免共线点
            jitter = 1e-6 * np.random.randn(*projected_points.shape)
            projected_points += jitter

            # 尝试三角剖分
            tri = Delaunay(projected_points)

            # 检查三角剖分结果是否有效
            if len(tri.simplices) == 0:
                print("错误：Delaunay三角剖分未生成任何三角形")
                self.mesh = None
                return False

            self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(selected_points)
            self.mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
            self.mesh.compute_vertex_normals()

            # 移除面积过小的三角形
            self.remove_small_triangles(area_threshold=1e-6)

            print(f"Delaunay三角剖分完成，三角面片数: {len(self.mesh.triangles)}")
            return True

        except Exception as e:
            print(f"Delaunay三角剖分失败: {e}")
            self.mesh = None  # 出错时设为None
            return False

    def _compute_density_batch(self, batch_indices, points, pcd_tree, knn):
        """计算点云密度的辅助函数（用于并行处理）"""
        batch_densities = np.zeros(len(batch_indices))
        for i, idx in enumerate(batch_indices):
            [_, idx_neighbors, _] = pcd_tree.search_knn_vector_3d(points[idx], knn)
            # 密度定义为到第k个最近邻的距离的倒数
            if len(idx_neighbors) > 1:
                batch_densities[i] = 1.0 / np.linalg.norm(points[idx] - points[idx_neighbors[-1]])
        return batch_indices, batch_densities

    def remove_small_triangles(self, area_threshold=1e-6):
        """移除面积过小的三角形"""
        if self.mesh is None or self.mesh.is_empty():
            return

        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        # 计算每个三角形的面积
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]

        # 计算三角形面积（叉积的模的一半）
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # 保留面积大于阈值的三角形
        valid_indices = areas > area_threshold
        valid_triangles = triangles[valid_indices]

        # 更新网格
        self.mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
        self.mesh.remove_unreferenced_vertices()
        self.mesh.compute_vertex_normals()

        print(f"移除了 {len(triangles) - len(valid_triangles)} 个小三角形，剩余 {len(valid_triangles)} 个三角形")

    def mesh_repair(self) -> None:
        """修复网格（健壮版本）"""
        if self.mesh is None or self.mesh.is_empty():
            print("警告：网格为空或未生成，无法修复")
            return

        print("修复网格...")
        start_time = time.time()

        # 检查网格基本属性
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        if len(vertices) < 3 or len(triangles) < 1:
            print("错误：网格顶点或面数不足，无法修复")
            return

        # Open3D 基础修复
        try:
            # 确保网格有顶点法线
            if not self.mesh.has_vertex_normals():
                self.mesh.compute_vertex_normals()

            # 移除孤立顶点
            self.mesh = self.mesh.remove_unreferenced_vertices()

            # 移除退化三角形
            self.mesh = self.mesh.remove_degenerate_triangles()

            # 尝试网格优化
            self.mesh = self.mesh.remove_non_manifold_edges()

            # 确保三角形朝向一致
            self.mesh.orient_triangles()

            # 重新计算法线
            self.mesh.compute_vertex_normals()

            print(f"Open3D 基础修复完成，面数: {len(self.mesh.triangles)}，耗时: {time.time() - start_time:.2f}秒")

        except Exception as e2:
            print(f"Open3D 修复失败: {e2}，使用原始网格")

    def mesh_simplification(self, target_faces: int = 100000) -> None:
        """简化网格（增强版）"""
        if self.mesh is None or self.mesh.is_empty():
            print("警告：网格为空或未生成，无法简化")
            return

        print(f"简化网格（目标面数: {target_faces}）...")
        start_time = time.time()

        try:
            # 检查当前面数
            current_faces = len(self.mesh.triangles)
            if current_faces <= target_faces:
                print(f"当前面数 ({current_faces}) 已小于或等于目标面数 ({target_faces})，跳过简化")
                return

            # 分阶段简化（避免一次性简化导致崩溃）
            if current_faces > 2 * target_faces:
                # 先简化50%，再达到目标面数
                mid_faces = int(current_faces * 0.5)
                self.mesh = self.mesh.simplify_quadric_decimation(target_number_of_triangles=mid_faces)
                print(f"第一阶段简化完成，当前面数: {len(self.mesh.triangles)}")

            self.mesh = self.mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

            # 优化简化后的网格
            self.mesh = self.mesh.remove_duplicated_vertices()
            self.mesh = self.mesh.remove_duplicated_triangles()
            self.mesh.compute_vertex_normals()

            print(f"网格简化完成，当前面数: {len(self.mesh.triangles)}，耗时: {time.time() - start_time:.2f}秒")

        except Exception as e:
            print(f"网格简化失败: {e}")
            print("回退到原始网格")

    def save_result(self) -> None:
        """保存处理结果"""
        if self.output_file is None:
            return

        print(f"保存结果到: {self.output_file}")
        start_time = time.time()

        try:
            if self.save_as_pcd and self.pcd:
                o3d.io.write_point_cloud(self.output_file, self.pcd)
            elif self.mesh:
                o3d.io.write_triangle_mesh(self.output_file, self.mesh)
            else:
                print("没有可保存的数据")
                return

            print(f"保存耗时: {time.time() - start_time:.2f}秒")

        except Exception as e:
            print(f"保存结果时出错: {e}")

    def visualize(self) -> None:
        """可视化结果"""
        if self.pcd is None and self.mesh is None:
            return

        geometries = []
        if self.pcd:
            geometries.append(self.pcd)
        if self.mesh:
            geometries.append(self.mesh)

        try:
            o3d.visualization.draw_geometries(geometries)
        except Exception as e:
            print(f"可视化时出错: {e}")
            print("尝试使用简单的可视化方法...")
            try:
                # 尝试简单的可视化方法
                if self.pcd:
                    o3d.visualization.draw_geometries([self.pcd])
                if self.mesh:
                    o3d.visualization.draw_geometries([self.mesh])
            except:
                print("所有可视化方法都失败")


def main():
    parser = argparse.ArgumentParser(description='点云/网格处理工具')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径(.obj, .ply, .stl)')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--reconstruction-method', '-rm', default='delaunay-vertical',
                        choices=['delaunay-vertical'],
                        help='网格重建方法')
    parser.add_argument('--density-threshold', type=float, default=0.3,
                        help='Delaunay-Vertical重建的密度阈值')
    parser.add_argument('--vertical-threshold', type=float, default=45.0,
                        help='Delaunay-Vertical重建的竖直阈值(度)')
    parser.add_argument('--knn', type=int, default=50,
                        help='用于计算密度和法线的最近邻数量')
    parser.add_argument('--voxel-size', type=float, default=0.01,
                        help='体素降采样的体素大小')
    parser.add_argument('--simplify', action='store_true', help='是否进行网格简化')
    parser.add_argument('--target-faces', type=int, default=100000, help='网格简化的目标面数')
    parser.add_argument('--save-as-pcd', action='store_true', help='保存为点云格式而非网格')

    args = parser.parse_args()

    # 初始化优化器
    optimizer = MeshOptimizer(args.input, args.output, args.save_as_pcd)

    # 加载数据
    if not optimizer.load_data():
        return

    if args.save_as_pcd:
        # 点云处理流程
        optimizer.preprocess_point_cloud(voxel_size=args.voxel_size)
    else:
        # 网格处理流程
        if optimizer.pcd:
            # 如果输入是点云，需要先重建网格
            print("从点云重建网格...")

            if args.reconstruction_method == 'delaunay-vertical':
                # Delaunay-Vertical重建
                optimizer.preprocess_point_cloud(
                    voxel_size=args.voxel_size,
                    target_reduction=0.8,  # 默认减少80%的点
                    voxel_size_start=0.01
                )

                success = optimizer.reconstruct_delaunay_vertical(
                    density_threshold=args.density_threshold,
                    vertical_threshold=args.vertical_threshold,
                    knn=args.knn
                )

                if not success:
                    print("Delaunay-Vertical重建失败")
                    return
            else:
                print(f"错误：不支持的重建方法: {args.reconstruction_method}")
                return

        # 修复网格
        if optimizer.mesh:
            optimizer.mesh_repair()

            # 简化网格
            if args.simplify:
                optimizer.mesh_simplification(target_faces=args.target_faces)
        else:
            print("错误：没有可处理的网格数据")
            return

    # 保存结果
    optimizer.save_result()

    # 可视化
    optimizer.visualize()


if __name__ == "__main__":
    main()
#python Deculary.py --input 2025_05_18_11_20_57.ply --output output.obj --reconstruction-method delaunay-vertical --simplify --target-faces 50000
#python step.py -i 2025_04_07_17_13_42.ply -o changfang.ply -rm delaunay-vertical --density-threshold 0.3 --vertical-threshold 40.0 --knn 20 --simplify --target-faces 5000000 --voxel-size 0.001
#a significant discrepancy