import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import open3d as o3d
from PIL import Image


class PointCloudProcessor:
    def __init__(self, file_path: str, downsample_voxel_size: float = 0.01):
        """初始化点云处理器"""
        self.file_path = file_path
        self.pcd = None
        self.downsample_voxel_size = downsample_voxel_size

    def load_point_cloud(self) -> bool:
        """加载并降采样点云数据"""
        try:
            # 读取点云
            self.pcd = o3d.io.read_point_cloud(self.file_path)

            if not self.pcd.has_points():
                print(f"错误：无法加载点云数据，文件可能为空或格式不支持：{self.file_path}")
                return False

            # 体素降采样
            if self.downsample_voxel_size > 0:
                self.pcd = self.pcd.voxel_down_sample(self.downsample_voxel_size)
                print(f"成功加载并降采样点云，点数: {len(self.pcd.points)}")
            else:
                print(f"成功加载点云，点数: {len(self.pcd.points)}")

            return True
        except Exception as e:
            print(f"加载点云时出错: {e}")
            return False

    def project_to_image(self, output_path: str, resolution: tuple = (1000, 1000),
                         color_mode: str = 'height', point_size: int = 2):
        """
        将点云投影为平面图像

        参数:
            output_path: 输出图像路径
            resolution: 图像分辨率 (宽度, 高度)
            color_mode: 着色模式，可选 'height'(高度)、'intensity'(强度)、'rgb'(RGB颜色)
            point_size: 点在图像中的大小（像素）
        """
        if self.pcd is None or not self.pcd.has_points():
            print("错误：没有加载点云数据")
            return False

        print("将点云投影为平面图像...")

        points = np.asarray(self.pcd.points)

        # 确保点云有颜色数据
        if color_mode == 'rgb' and not self.pcd.has_colors():
            print("警告：点云没有RGB颜色数据，将使用高度着色")
            color_mode = 'height'

        # 如果使用强度着色，但点云没有颜色，则使用高度
        if color_mode == 'intensity' and not self.pcd.has_colors():
            print("警告：点云没有强度数据，将使用高度着色")
            color_mode = 'height'

        # 获取颜色数据
        if color_mode == 'rgb':
            colors = np.asarray(self.pcd.colors)
        elif color_mode == 'intensity' and self.pcd.has_colors():
            # 使用RGB的平均值作为强度
            colors = np.asarray(self.pcd.colors)
            intensity = np.mean(colors, axis=1)
            colors = np.stack([intensity, intensity, intensity], axis=1)
        else:  # 默认使用高度着色
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])
            # 归一化高度值到[0,1]范围
            heights = (points[:, 2] - z_min) / (z_max - z_min + 1e-10)
            # 创建从蓝色到红色的颜色映射
            colors = np.zeros((len(points), 3))
            colors[:, 0] = heights  # R
            colors[:, 2] = 1 - heights  # B

        # 计算点云边界
        x_min, y_min, _ = np.min(points, axis=0)
        x_max, y_max, _ = np.max(points, axis=0)

        # 创建空白图像
        image = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255  # 白色背景

        # 计算缩放因子
        scale_x = resolution[0] / (x_max - x_min)
        scale_y = resolution[1] / (y_max - y_min)
        scale = min(scale_x, scale_y)

        # 将点投影到图像
        for i in range(len(points)):
            x, y = points[i, 0], points[i, 1]

            # 转换为图像坐标
            img_x = int((x - x_min) * scale)
            img_y = int((y_max - y) * scale)  # 注意：图像的y轴是向下的

            # 确保坐标在图像范围内
            if 0 <= img_x < resolution[0] and 0 <= img_y < resolution[1]:
                # 设置点的颜色
                r, g, b = colors[i]
                r = int(r * 255)
                g = int(g * 255)
                b = int(b * 255)

                # 考虑点大小
                for dx in range(-point_size, point_size + 1):
                    for dy in range(-point_size, point_size + 1):
                        nx = img_x + dx
                        ny = img_y + dy
                        if 0 <= nx < resolution[0] and 0 <= ny < resolution[1]:
                            # 简单的距离衰减，使点看起来更自然
                            distance = np.sqrt(dx * dx + dy * dy)
                            if distance <= point_size:
                                alpha = 1.0 - (distance / point_size)
                                image[ny, nx] = (
                                    int(image[ny, nx, 0] * (1 - alpha) + r * alpha),
                                    int(image[ny, nx, 1] * (1 - alpha) + g * alpha),
                                    int(image[ny, nx, 2] * (1 - alpha) + b * alpha)
                                )

        # 保存图像
        img = Image.fromarray(image)
        img.save(output_path)

        print(f"点云投影平面图像已保存至: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description='点云投影平面图像生成工具')
    parser.add_argument('--input', '-i', default='input.ply', help='输入点云文件路径')
    parser.add_argument('--output', '-o', default='point_cloud_image.png', help='输出图像文件路径')
    parser.add_argument('--voxel-size', type=float, default=0.01, help='体素降采样大小')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1000, 1000], help='图像分辨率 (宽度 高度)')
    parser.add_argument('--color-mode', choices=['height', 'intensity', 'rgb'], default='height', help='着色模式')
    parser.add_argument('--point-size', type=int, default=2, help='点在图像中的大小（像素）')

    args = parser.parse_args()

    # 处理点云
    processor = PointCloudProcessor(
        args.input,
        downsample_voxel_size=args.voxel_size
    )

    if processor.load_point_cloud():
        processor.project_to_image(
            output_path=args.output,
            resolution=tuple(args.resolution),
            color_mode=args.color_mode,
            point_size=args.point_size
        )
        print(f"点云投影平面图像已保存至: {args.output}")


if __name__ == "__main__":
    main()









