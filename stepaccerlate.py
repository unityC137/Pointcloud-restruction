import numpy as np
import open3d as o3d
import concurrent.futures
from scipy.spatial import Delaunay
from typing import Optional
import argparse
import time

def _compute_density_batch(batch_indices, points, knn):
    """计算点云密度的辅助函数（用于并行处理）"""
    batch_densities = np.zeros(len(batch_indices))
    pcd_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points)))
    for i, idx in enumerate(batch_indices):
        [_, idx_neighbors, _] = pcd_tree.search_knn_vector_3d(points[idx], knn)
        # 密度定义为到第k个最近邻的距离的倒数
        if len(idx_neighbors) > 1:
            batch_densities[i] = 1.0 / np.linalg.norm(points[idx] - points[idx_neighbors[-1]])
    return batch_indices, batch_densities

class MeshOptimizer:
    def __init__(self, input_file: str, output_file: Optional[str] = None, save_as_pcd: bool = False):
        self.input_file = input_file
        self.output_file = output_file
        self.save_as_pcd = save_as_pcd
        self.pcd = None
        self.mesh = None
        self.num_threads = 10  # 默认线程数

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
                print(f"加载耗时: {time.time() - start_time:.2f}秒")
                return True

            elif self.input_file.lower().endswith(('.obj', '.stl', '.off', '.gltf', '.glb')):
                self.mesh = o3d.io.read_triangle_mesh(self.input_file)
                if not self.mesh.has_vertices():
                    print(f"错误: 文件 {self.input_file} 不包含顶点数据")
                    return False

                print(f"成功加载网格，顶点数: {len(self.mesh.vertices)}, 面数: {len(self.mesh.triangles)}")

                # 如果需要保存为点云，从网格提取点云
                if self.save_as_pcd:
                    self.pcd = o3d.geometry.PointCloud()
                    self.pcd.points = self.mesh.vertices
                    if self.mesh.has_vertex_normals():
                        self.pcd.normals = self.mesh.vertex_normals

                    print(f"已从网格提取点云，点数: {len(self.pcd.points)}")

                print(f"加载耗时: {time.time() - start_time:.2f}秒")
                return True

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
        pcd = pcd.copy()
        original_points = len(pcd.points)
        current_reduction = 0.0

        # 阶段1：体素降采样（逐步缩小体素大小）
        voxel_size = voxel_size_start
        for _ in range(max_iter):
            if len(pcd.points) <= original_points * (1 - target_reduction):
                break
            pcd = pcd.voxel_down_sample(voxel_size)
            current_reduction = 1 - len(pcd.points) / original_points
            voxel_size *= 0.5  # 每次尝试减半体素大小
            if voxel_size < 1e-5:  # 防止体素过小
                break

        #   stage 2 random downsampleing (ensuring the target number of points is reached)
        if current_reduction < target_reduction:
            target_points = int(original_points * (1 - target_reduction))
            if target_points > 0:
                pcd = pcd.uniform_down_sample(every_k_points=len(pcd.points) // target_points + 1)

        print(
            f"降采样完成，点数从 {original_points} 减少到 {len(pcd.points)} ({current_reduction * 100:.2f}% reduction)")
        return pcd

    def preprocess_point_cloud(self, voxel_size: float = 0.005, target_reduction=0.9, voxel_size_start=0.01,
                               max_iter=5) -> None:
        """预处理点云数据"""
        if self.pcd is None:
            print("警告：点云数据为空，无法进行预处理")
            return

        print(f"预处理点云 (体素大小: {voxel_size})")
        start_time = time.time()

        try:
            # ADAPTIVE DOWNSAMPLING
            self.pcd = self.adaptive_downsample(self.pcd, target_reduction=target_reduction,
                                                voxel_size_start=voxel_size_start, max_iter=max_iter)

            # Estimate the normal vector
            if not self.pcd.has_normals():
                print("估计点云法线...")
                self.pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                # 法线定向
                self.pcd.orient_normals_consistent_tangent_plane(100)

            print(f"点云预处理完成，耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            # print(f"点云预处理出错: {e}")
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
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
            )
            # 法线定向
            self.pcd.orient_normals_consistent_tangent_plane(knn)

        points = np.asarray(self.pcd.points)
        normals = np.asarray(self.pcd.normals)

        # 1. 多线程计算点云密度
        print(f"计算点云密度（使用{self.num_threads}个线程）...")
        densities = np.zeros(len(points))
        batch_size = max(1, len(points) // self.num_threads)
        batches = [range(i, min(i + batch_size, len(points))) for i in range(0, len(points), batch_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_batch = {executor.submit(_compute_density_batch, batch, points, knn): batch for batch in batches}
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_indices, batch_densities = future.result()
                densities[batch_indices] = batch_densities

        # 归一化密度
        if np.max(densities) > 0:
            densities = densities / np.max(densities)

        # 2. 识别竖直区域
        vertical_direction = np.array([0, 0, 1])
        normal_angles = np.arccos(np.abs(np.dot(normals, vertical_direction))) * 180 / np.pi
        dense_mask = densities > density_threshold
        vertical_mask = normal_angles < vertical_threshold
        selected_mask = np.logical_and(dense_mask, vertical_mask)
        selected_indices = np.where(selected_mask)[0]

        print(f"找到 {len(selected_indices)} 个点符合密集且竖直的条件")
        if len(selected_indices) < 3:
            print("错误：符合条件的点太少，无法进行Delaunay三角剖分")
            self.mesh = None  # 明确设置为None
            return False

        selected_points = points[selected_indices]

        # 3. 执行Delaunay三角剖分（增加异常处理）
        try:
            projected_points = selected_points[:, :2]
            tri = Delaunay(projected_points)

            # 检查三角剖分结果是否有效
            if len(tri.simplices) == 0:
                # print("错误：Delaunay三角剖分未生成任何三角形")
                self.mesh = None
                return False

            self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(selected_points)
            self.mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
            self.mesh.compute_vertex_normals()
            # print(f"Delaunay三角剖分完成，三角面片数: {len(self.mesh.triangles)}")
            return True

        except Exception as e:
            # print(f"Delaunay三角剖分失败: {e}")
            self.mesh = None  # 出错时设为None
            return False

    def mesh_repair(self) -> None:
        """修复网格（健壮版本）"""
        if self.mesh is None or self.mesh.is_empty():
            # print("警告：网格为空或未生成，无法修复")
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
    #
    # def visualize(self) -> None:
    #     """可视化结果"""
    #     if self.pcd is None and self.mesh is None:
    #         return
    #
    #     geometries = []
    #     if self.pcd:
    #         geometries.append(self.pcd)
    #     if self.mesh:
    #         geometries.append(self.mesh)
    #
    #     try:



    #         o3d.visualization.draw_geometries(geometries)
    #     except Exception as e:
    #         print(f"可视化时出错: {e}")
    #         print("尝试使用简单的可视化方法...")
    #         try:
    #             # 尝试简单的可视化方法
    #             if self.pcd:
    #                 o3d.visualization.draw_geometries([self.pcd])
    #             if self.mesh:
    #                 o3d.visualization.draw_geometries([self.mesh])
    #         except:
    #             print("所有可视化方法都失败")
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
    parser.add_argument('--density-threshold', type=float, default=0.5,
                        help='Delaunay-Vertical重建的密度阈值')
    parser.add_argument('--vertical-threshold', type=float, default=30.0,
                        help='Delaunay-Vertical重建的竖直阈值(度)')
    parser.add_argument('--knn', type=int, default=30,
                        help='用于计算密度和法线的最近邻数量')
    parser.add_argument('--voxel-size', type=float, default=0.005,
                        help='体素降采样的体素大小')
    parser.add_argument('--simplify', action='store_true', help='是否进行网格简化')
    parser.add_argument('--target-faces', type=int, default=100000, help='网格简化的目标面数')
    parser.add_argument('--save-as-pcd', action='store_true', help='保存为点云格式而非网格')
    parser.add_argument('--visualize', action='store_true', help='是否进行可视化')  # 新增参数

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

    # # 可视化
    # optimizer.visualize()
    if args.visualize:
        optimizer.visualize()

if __name__ == "__main__":
    main()