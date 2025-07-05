import open3d as o3d
# # 1. 读取PLY点云
# pcd = o3d.io.read_point_cloud("output_repaired1.ply")
# print(pcd)  # 查看点云信息(点数等)
##
import pyvista as pv

# 1. 读取 OBJ 文件
mesh = pv.read("cuboid_mesh.obj")

# 2. 创建一个绘图器并添加模型
plotter = pv.Plotter()
plotter.add_mesh(mesh, color=True, show_edges=True)
plotter.add_axes()
plotter.show(title="OBJ 文件显示")
