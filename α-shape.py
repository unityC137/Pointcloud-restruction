import bpy
import os

# 清除默认场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 导入Open3D生成的网格
obj_path = "reconstructed_mesh.obj"
bpy.ops.import_scene.obj(filepath=obj_path)
model = bpy.context.selected_objects[0]

# 创建生锈材质
material = bpy.data.materials.new(name="RustMaterial")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links

# 清除默认节点
for node in nodes:
    nodes.remove(node)

# 创建Principled BSDF节点
bsdf = nodes.new('ShaderNodeBsdfPrincipled')
bsdf.inputs['Base Color'].default_value = (0.6, 0.3, 0.2, 1.0)  # 锈红色
bsdf.inputs['Metallic'].default_value = 0.4  # 部分金属特性
bsdf.inputs['Roughness'].default_value = 0.8  # 粗糙表面

# 添加噪点纹理以模拟锈迹变化
noise_texture = nodes.new('ShaderNodeTexNoise')
noise_texture.inputs['Scale'].default_value = 10.0  # 噪声尺度
noise_texture.inputs['Detail'].default_value = 4.0  # 细节级别

# 添加颜色渐变以控制锈蚀程度
color_ramp = nodes.new('ShaderNodeValToRGB')
color_ramp.color_ramp.elements[0].color = (0.8, 0.8, 0.8, 1.0)  # 未锈蚀区域（灰色）
color_ramp.color_ramp.elements[1].color = (0.6, 0.3, 0.2, 1.0)  # 锈蚀区域（红棕色）

# 添加材质输出节点
output = nodes.new('ShaderNodeOutputMaterial')

# 连接节点
links.new(noise_texture.outputs['Fac'], color_ramp.inputs['Fac'])
links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# 将材质赋给模型
if model.data.materials:
    model.data.materials[0] = material
else:
    model.data.materials.append(material)

# 设置光源
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 5.0

# 设置相机
bpy.ops.object.camera_add(location=(10, 10, 10))
camera = bpy.context.active_object
bpy.context.scene.camera = camera
camera.location = (10, 10, 10)
camera.rotation_euler = (0.7, 0, 0.785)

# 设置渲染引擎为Cycles
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'  # 使用GPU加速

# 设置输出路径
output_path = os.path.dirname(obj_path) + "/rust_render.png"
bpy.context.scene.render.filepath = output_path

# 渲染
bpy.ops.render.render(write_still=True)