"""SMPL-H / SMPL-X motion (.npz) -> skinned, animated FBX.

First stage of the pipeline:

    motion .npz  --(this script)-->  .fbx  --> camera trajectory --> render

`render_pipeline.py` calls this automatically when a motion has no .fbx
next to it. It can also be run standalone to pre-convert motions.

The FBX is built from scratch with the Autodesk FBX SDK:
  - a template mesh shaped by `betas` (optionally with UVs from an OBJ
    template, needed only when skin textures are used)
  - a skeleton whose rest offsets come from the shaped joints
  - linear-blend-skinning weights from the body model
  - a bind pose and per-frame local rotation / root translation keyframes

The joint layout is taken from the body model: a 52-joint model gives
SMPL-H, a 55-joint model gives SMPL-X. The motion's `poses` must have the
matching size (F, 156) or (F, 165).

Motion .npz keys: poses (F, J*3), betas (B,) or (1, B), trans (F, 3),
optional mocap_framerate (defaults to 30).

Requirements: the `fbx` Python module (Autodesk FBX SDK Python bindings).

Usage:
    python scripts/smplh2fbx.py \
        --model_path body_models/smplh/neutral/model.npz \
        --input_path motion.npz \
        --output_path motion.fbx \
        [--obj_template body_models/smplh_uv.obj]
"""

import os
import glob
import pickle
import shutil
import argparse
import tempfile

import numpy as np
import torch
from transforms3d.euler import mat2euler

import fbx


# FBX bones are created in this order (index == body-model joint index)
SMPLH_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]  # 52 joints

# SMPL-X inserts Jaw / eyes after the 22 body joints
SMPLX_JOINT_NAMES = SMPLH_JOINT_NAMES[:22] + ["Jaw", "L_Eye", "R_Eye"] + SMPLH_JOINT_NAMES[22:]  # 55 joints


# ---------------------------------------------------------------------------
# body model loading
# ---------------------------------------------------------------------------

def _to_dense(x):
    if 'scipy.sparse' in str(type(x)):
        x = x.todense()
    return np.asarray(x)


def load_body_model(model_path):
    """Load the arrays needed for FBX export from a SMPL-H/X .npz or .pkl."""
    if model_path.endswith('.npz'):
        data = dict(np.load(model_path, allow_pickle=True))
    elif model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    else:
        raise ValueError(f'unsupported body model format: {model_path}')

    parents = _to_dense(data['kintree_table'])[0].astype(np.int64)
    parents[0] = -1

    return {
        'v_template': _to_dense(data['v_template']).astype(np.float32),
        'shapedirs': _to_dense(data['shapedirs']).astype(np.float32),
        'J_regressor': _to_dense(data['J_regressor']).astype(np.float32),
        'weights': _to_dense(data['weights']).astype(np.float32),
        'faces': _to_dense(data['f']).astype(np.int64),
        'parents': parents,
    }


def joint_names_for(num_joints):
    if num_joints == len(SMPLH_JOINT_NAMES):
        return SMPLH_JOINT_NAMES
    if num_joints == len(SMPLX_JOINT_NAMES):
        return SMPLX_JOINT_NAMES
    raise ValueError(f'unsupported body model with {num_joints} joints (expected 52 SMPL-H or 55 SMPL-X)')


def read_uv(obj_template):
    """Read UVs from an OBJ template.

    Returns (uv_coords, uv_faces, vert_faces), or (None, None, None) if absent.
    vert_faces holds the vertex indices of each face, used to match faces with the model.
    """
    if not obj_template:
        return None, None, None
    if not os.path.isfile(obj_template):
        print(f'warning: OBJ template not found, exporting without UVs: {obj_template}')
        return None, None, None

    uv_coords, uv_faces, vert_faces = [], [], []
    with open(obj_template, 'r') as f:
        for line in f:
            if line.startswith('vt '):
                parts = line.split()
                uv_coords.append([float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                face_vs, face_uvs = [], []
                for part in line.split()[1:]:
                    idx = part.split('/')
                    face_vs.append(int(idx[0]) - 1)  # OBJ is 1-based
                    if len(idx) > 1 and idx[1]:
                        face_uvs.append(int(idx[1]) - 1)
                if len(face_uvs) == 3:
                    uv_faces.append(face_uvs)
                    vert_faces.append(face_vs)
    print(f'loaded {len(uv_coords)} UV coordinates and {len(uv_faces)} UV faces from {obj_template}')
    return np.array(uv_coords), np.array(uv_faces), np.array(vert_faces)


def match_uv_faces(model_faces, uv_faces, vert_faces):
    """Reorder OBJ UV faces to follow the model's face order.

    Faces are matched by their vertex indices, so the OBJ may list faces in any
    order or start each triangle at a different corner. The OBJ must use the
    model's vertex order (true for meshes exported from the SMPL-H FBX).
    """
    if len(vert_faces) != len(model_faces):
        raise ValueError(f'UV template has {len(vert_faces)} triangles, the body model has '
                         f'{len(model_faces)}; export the OBJ with triangulated faces')
    lookup = {}
    for vs, uvs in zip(vert_faces, uv_faces):
        for k in range(3):  # all rotations, winding order kept
            lookup[tuple(np.roll(vs, -k))] = np.roll(uvs, -k)
    out = []
    for face in model_faces:
        uvs = lookup.get(tuple(face))
        if uvs is None:
            raise ValueError('UV template faces do not match the body model; the OBJ must keep '
                             'the SMPL-H vertex order (export the SMPL-H FBX mesh unchanged)')
        out.append(uvs)
    return np.array(out)


# ---------------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------------

def axis_angle_to_rotmat(theta):
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (via quaternions)."""
    shape = theta.shape[:-1]
    flat = theta.reshape(-1, 3)
    angle = torch.norm(flat + 1e-8, p=2, dim=1, keepdim=True)
    axis = flat / angle
    half = angle * 0.5
    quat = torch.cat([torch.cos(half), torch.sin(half) * axis], dim=1)
    quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    rot = torch.stack([
        w2 + x2 - y2 - z2, 2 * (x * y - w * z), 2 * (w * y + x * z),
        2 * (w * z + x * y), w2 - x2 + y2 - z2, 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (w * x + y * z), w2 - x2 - y2 + z2,
    ], dim=1)
    return rot.view(*shape, 3, 3)


def compute_skeleton(model, betas, poses, trans, scale):
    """Shaped rest mesh, per-joint rest offsets and per-frame local transforms.

    Returns:
        v_shaped (V, 3), offsets (J, 3), local_rot (F, J, 3, 3), root_trans (F, 3)
        all lengths multiplied by `scale`.
    """
    num_betas = min(betas.shape[-1], model['shapedirs'].shape[-1])
    betas = betas.reshape(-1)[:num_betas]
    shapedirs = model['shapedirs'][:, :, :num_betas]

    v_shaped = model['v_template'] + np.einsum('l,mkl->mk', betas, shapedirs)
    joints = model['J_regressor'] @ v_shaped

    parents = model['parents']
    offsets = joints.copy()
    offsets[1:] -= joints[parents[1:]]

    local_rot = axis_angle_to_rotmat(torch.from_numpy(poses).float()).numpy()
    root_trans = offsets[0][None] + trans

    return v_shaped * scale, offsets * scale, local_rot, root_trans * scale


# ---------------------------------------------------------------------------
# FBX scene construction
# ---------------------------------------------------------------------------

def add_mesh(scene, vertices, faces, uv_coords=None, uv_faces=None):
    node = fbx.FbxNode.Create(scene, "Geometry")
    scene.GetRootNode().AddChild(node)

    mesh = fbx.FbxMesh.Create(scene, "body")
    node.SetNodeAttribute(mesh)

    mesh.InitControlPoints(vertices.shape[0])
    for i, v in enumerate(vertices):
        mesh.SetControlPointAt(fbx.FbxVector4(float(v[0]), float(v[1]), float(v[2])), i)

    for i, face in enumerate(faces):
        mesh.BeginPolygon(i)
        for vid in face:
            mesh.AddPolygon(int(vid))
        mesh.EndPolygon()

    if uv_coords is not None and uv_faces is not None and len(uv_faces) == len(faces):
        uv_layer = mesh.CreateElementUV("UVSet")
        uv_layer.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
        uv_layer.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
        direct = uv_layer.GetDirectArray()
        for uv in uv_coords:
            direct.Add(fbx.FbxVector2(float(uv[0]), float(uv[1])))
        index = uv_layer.GetIndexArray()
        for face_uv in uv_faces:
            for uid in face_uv:
                index.Add(int(uid))

    return node


def add_skeleton(manager, scene, offsets, parents, joint_names):
    reference = fbx.FbxNode.Create(scene, "Reference")
    scene.GetRootNode().AddChild(reference)

    nodes = []
    for j, name in enumerate(joint_names):
        skeleton = fbx.FbxSkeleton.Create(manager, "")
        skeleton.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)
        node = fbx.FbxNode.Create(scene, name)
        node.SetNodeAttribute(skeleton)
        node.LclTranslation.Set(fbx.FbxDouble3(*[float(x) for x in offsets[j]]))
        nodes.append(node)
        if parents[j] != -1:
            nodes[parents[j]].AddChild(node)

    reference.AddChild(nodes[0])
    return nodes


def add_skin(scene, weights, geometry_node, skeleton_nodes):
    evaluator = scene.GetAnimationEvaluator()
    geometry_matrix = evaluator.GetNodeGlobalTransform(geometry_node)

    skin = fbx.FbxSkin.Create(scene, "")
    for j, bone in enumerate(skeleton_nodes):
        cluster = fbx.FbxCluster.Create(scene, "")
        cluster.SetLink(bone)
        cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eTotalOne)
        for vid in np.nonzero(weights[:, j] > 0)[0]:
            cluster.AddControlPointIndex(int(vid), float(weights[vid, j]))
        cluster.SetTransformMatrix(geometry_matrix)
        cluster.SetTransformLinkMatrix(evaluator.GetNodeGlobalTransform(bone))
        skin.AddCluster(cluster)

    geometry_node.GetNodeAttribute().AddDeformer(skin)


def store_bind_pose(scene, geometry_node, skeleton_nodes):
    """Store the bind pose: all bones (and their parents) plus the mesh."""
    pose = fbx.FbxPose.Create(scene, geometry_node.GetName())
    pose.SetIsBindPose(True)

    added = []

    def add_with_parents(node):
        if not node or node in added:
            return
        add_with_parents(node.GetParent())
        added.append(node)

    for bone in skeleton_nodes:
        add_with_parents(bone)
    added.append(geometry_node)

    evaluator = scene.GetAnimationEvaluator()
    for node in added:
        pose.Add(node, fbx.FbxMatrix(evaluator.GetNodeGlobalTransform(node)))
    scene.AddPose(pose)


def _key_channels(layer, prop, values, frame_duration):
    """Write per-frame X/Y/Z keys (constant interpolation) for one property."""
    time = fbx.FbxTime()
    for axis, name in enumerate("XYZ"):
        curve = prop.GetCurve(layer, name, True)
        curve.KeyModifyBegin()
        for f, value in enumerate(values):
            time.SetSecondDouble(f * frame_duration)
            key = curve.KeyAdd(time)[0]
            curve.KeySetValue(key, float(value[axis]))
            # cubic interpolation produces artifacts; keys exist on every frame anyway
            curve.KeySetInterpolation(key, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
        curve.KeyModifyEnd()


def animate(scene, skeleton_nodes, local_rot, root_trans, fps):
    frame_duration = 1.0 / fps
    stack = fbx.FbxAnimStack.Create(scene, "Take1")
    layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
    stack.AddMember(layer)

    _key_channels(layer, skeleton_nodes[0].LclTranslation, root_trans, frame_duration)

    for j, node in enumerate(skeleton_nodes):
        eulers = [np.rad2deg(mat2euler(local_rot[f, j], axes="sxyz")) for f in range(local_rot.shape[0])]
        _key_channels(layer, node.LclRotation, eulers, frame_duration)


def save_scene(path, manager, scene):
    exporter = fbx.FbxExporter.Create(manager, "")
    if not exporter.Initialize(path):
        raise RuntimeError(f"FBX exporter failed to initialize: {exporter.GetStatus().GetErrorString()}")
    exporter.Export(scene)
    exporter.Destroy()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

class SMPLH2FBX:
    """Loads the body model (and optional UV template) once, converts many motions."""

    def __init__(self, model_path, obj_template=None, scale=100.0):
        """
        Args:
            model_path: SMPL-H (52 joints) or SMPL-X (55 joints) model, .npz or .pkl
            obj_template: optional OBJ with UVs matching the model topology
            scale: unit scale; SMPL is in meters, the FBX is written in cm
        """
        print(f'[SMPLH2FBX] loading body model: {model_path}')
        self.model = load_body_model(model_path)
        self.joint_names = joint_names_for(len(self.model['parents']))
        self.uv_coords, self.uv_faces, vert_faces = read_uv(obj_template)
        if self.uv_faces is not None:
            self.uv_faces = match_uv_faces(self.model['faces'], self.uv_faces, vert_faces)
        self.scale = scale

    def convert(self, npz_path, fbx_path):
        """Convert one motion .npz to .fbx. Returns True on success."""
        data = dict(np.load(npz_path, allow_pickle=True))
        num_joints = len(self.joint_names)

        poses = data['poses'].reshape(data['poses'].shape[0], -1, 3)
        if poses.shape[1] != num_joints:
            raise ValueError(f'{npz_path}: poses have {poses.shape[1]} joints, '
                             f'body model has {num_joints}')
        betas = np.asarray(data['betas'], dtype=np.float32)
        if betas.ndim == 2:
            betas = betas[0]
        trans = np.asarray(data['trans'], dtype=np.float32)
        fps = float(data['mocap_framerate']) if 'mocap_framerate' in data else 30.0

        v_shaped, offsets, local_rot, root_trans = compute_skeleton(
            self.model, betas, poses.astype(np.float32), trans, self.scale)

        manager = fbx.FbxManager.Create()
        scene = fbx.FbxScene.Create(manager, "")
        scene.GetGlobalSettings().SetTimeMode(fbx.FbxTime().ConvertFrameRateToTimeMode(fps))

        geometry = add_mesh(scene, v_shaped, self.model['faces'], self.uv_coords, self.uv_faces)
        bones = add_skeleton(manager, scene, offsets, self.model['parents'], self.joint_names)
        add_skin(scene, self.model['weights'], geometry, bones)
        store_bind_pose(scene, geometry, bones)
        animate(scene, bones, local_rot, root_trans, fps)

        # write to a temp file first so a crash never leaves a corrupt .fbx behind
        out_dir = os.path.dirname(os.path.abspath(fbx_path))
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix='.fbx', dir=out_dir)
        os.close(fd)
        try:
            save_scene(tmp_path, manager, scene)
            shutil.move(tmp_path, fbx_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            manager.Destroy()

        return os.path.exists(fbx_path)


def parse_args():
    parser = argparse.ArgumentParser(description="SMPL-H / SMPL-X motion npz -> FBX")
    parser.add_argument('--model_path', type=str, default='body_models/smplh/neutral/model.npz',
                        help='SMPL-H (52 joints) or SMPL-X (55 joints) body model, .npz or .pkl')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='motion .npz file, or a directory of .npz files')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                        help='output .fbx (file input) or directory (directory input); '
                             'defaults to next to the input')
    parser.add_argument('--obj_template', type=str, default=None,
                        help='optional OBJ template with UVs (needed only for skin textures)')
    parser.add_argument('--scale', type=float, default=100.0,
                        help='unit scale (SMPL meters -> FBX centimeters)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    converter = SMPLH2FBX(args.model_path, obj_template=args.obj_template, scale=args.scale)

    if os.path.isdir(args.input_path):
        inputs = sorted(glob.glob(os.path.join(args.input_path, '*.npz')))
        out_dir = args.output_path or args.input_path
        jobs = [(p, os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + '.fbx')) for p in inputs]
    else:
        out = args.output_path or os.path.splitext(args.input_path)[0] + '.fbx'
        jobs = [(args.input_path, out)]

    failed = 0
    for npz_path, fbx_path in jobs:
        ok = converter.convert(npz_path, fbx_path)
        print(f"{'[ok]' if ok else '[failed]'} {npz_path} -> {fbx_path}")
        failed += 0 if ok else 1
    raise SystemExit(1 if failed else 0)
