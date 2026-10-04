"""Export the SMPL-H mesh with UVs from the MANO SMPL-H FBX to an OBJ (run inside Blender).

Usage:
    blender --background --python scripts/export_uv_obj.py -- f_avg_noFlatHand.fbx body_models/smplh_uv.obj

Usually called through scripts/setup_body_models.sh --uv_fbx.
"""

import sys

import bpy

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
if len(argv) != 2:
    print('usage: blender --background --python export_uv_obj.py -- <in.fbx> <out.obj>')
    sys.exit(1)
fbx_path, obj_path = argv

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if len(meshes) != 1:
    print(f'error: expected one mesh in {fbx_path}, found {len(meshes)}')
    sys.exit(1)
mesh = meshes[0]
if not mesh.data.uv_layers:
    print(f'error: mesh {mesh.name} has no UV layer')
    sys.exit(1)
print(f'mesh {mesh.name}: {len(mesh.data.vertices)} vertices, {len(mesh.data.polygons)} faces')

bpy.ops.object.select_all(action='DESELECT')
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh

bpy.ops.wm.obj_export(
    filepath=obj_path,
    export_selected_objects=True,
    export_uv=True,
    export_normals=False,
    export_materials=False,
    export_triangulated_mesh=True,
    apply_modifiers=False,
)
print(f'exported {obj_path}')
