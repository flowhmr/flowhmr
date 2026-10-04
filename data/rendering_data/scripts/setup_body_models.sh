#!/usr/bin/env bash
# Prepare body_models/ for rendering.
#
# Usage (from data/rendering_data/):
#   bash scripts/setup_body_models.sh [MODEL] [--uv_fbx FBX [--blender BLENDER] | --uv_obj OBJ]
#
#   MODEL      SMPL-H neutral model, .npz (linked) or .pkl (converted to .npz).
#              Default: ../../assets/body_models/smplh/neutral/model.npz (FlowHMR repo)
#   --uv_fbx   SMPL-H FBX from the MANO downloads (e.g. f_avg_noFlatHand.fbx); its mesh
#              and UVs are exported to body_models/smplh_uv.obj with Blender.
#   --blender  Blender executable for --uv_fbx (default: blender)
#   --uv_obj   an already exported OBJ with SMPL-H UVs, linked instead.
#
# Result:
#   body_models/smplh/neutral/model.npz
#   body_models/smplh_uv.obj            (with --uv_fbx or --uv_obj)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=""
UV_OBJ=""
UV_FBX=""
BLENDER="blender"
while [ $# -gt 0 ]; do
    case "$1" in
        --uv_obj) UV_OBJ="$2"; shift 2 ;;
        --uv_fbx) UV_FBX="$2"; shift 2 ;;
        --blender) BLENDER="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) MODEL="$1"; shift ;;
    esac
done
MODEL="${MODEL:-${ROOT}/../../assets/body_models/smplh/neutral/model.npz}"

abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }

# SMPL-H body model
DST_DIR="${ROOT}/body_models/smplh/neutral"
DST="${DST_DIR}/model.npz"
if [ ! -f "${MODEL}" ]; then
    echo "SMPL-H model not found: ${MODEL}" >&2
    echo "Prepare the FlowHMR body models first (docs/install.md, section 4.1)," >&2
    echo "or pass the path to a SMPL-H neutral .npz / .pkl." >&2
    exit 1
fi
MODEL="$(abspath "${MODEL}")"
mkdir -p "${DST_DIR}"
rm -f "${DST}"
case "${MODEL}" in
    *.npz)
        ln -s "${MODEL}" "${DST}"
        echo "linked ${DST} -> ${MODEL}"
        ;;
    *.pkl)
        python - "${MODEL}" "${DST}" <<'EOF'
import pickle
import sys

import numpy as np

src, dst = sys.argv[1:3]
with open(src, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
keys = ['v_template', 'shapedirs', 'posedirs', 'J_regressor', 'weights', 'kintree_table', 'f']
out = {}
for k in keys:
    v = data[k]
    if 'scipy.sparse' in str(type(v)):
        v = v.toarray()
    out[k] = np.asarray(v)
np.savez(dst, **out)
EOF
        echo "converted ${MODEL} -> ${DST}"
        ;;
    *)
        echo "unsupported model format: ${MODEL} (expected .npz or .pkl)" >&2
        exit 1
        ;;
esac

# UV template for skin textures
UV_DST="${ROOT}/body_models/smplh_uv.obj"

check_uv_obj() {
    local num_vt num_f
    num_vt=$(grep -c '^vt ' "$1" || true)
    num_f=$(grep -c '^f ' "$1" || true)
    if [ "${num_vt}" -eq 0 ] || [ "${num_f}" -ne 13776 ]; then
        echo "invalid UV OBJ: $1 has ${num_vt} UVs and ${num_f} faces" >&2
        echo "(expected UVs and 13776 triangles)" >&2
        exit 1
    fi
}

if [ -n "${UV_FBX}" ] && [ -n "${UV_OBJ}" ]; then
    echo "pass either --uv_fbx or --uv_obj, not both" >&2
    exit 1
elif [ -n "${UV_FBX}" ]; then
    if [ ! -f "${UV_FBX}" ]; then
        echo "SMPL-H FBX not found: ${UV_FBX}" >&2
        exit 1
    fi
    rm -f "${UV_DST}"
    "${BLENDER}" --background --python "${ROOT}/scripts/export_uv_obj.py" -- \
        "$(abspath "${UV_FBX}")" "${UV_DST}" > "${ROOT}/body_models/export_uv_obj.log" 2>&1 || true
    if [ ! -f "${UV_DST}" ]; then
        echo "UV export failed, see body_models/export_uv_obj.log" >&2
        exit 1
    fi
    check_uv_obj "${UV_DST}"
    echo "exported ${UV_DST} from ${UV_FBX}"
elif [ -n "${UV_OBJ}" ]; then
    if [ ! -f "${UV_OBJ}" ]; then
        echo "UV OBJ not found: ${UV_OBJ}" >&2
        exit 1
    fi
    check_uv_obj "${UV_OBJ}"
    UV_OBJ="$(abspath "${UV_OBJ}")"
    rm -f "${UV_DST}"
    ln -s "${UV_OBJ}" "${UV_DST}"
    echo "linked ${UV_DST} -> ${UV_OBJ}"
elif [ ! -e "${UV_DST}" ]; then
    echo "note: ${UV_DST} is missing; skin textures need it (pass --uv_fbx, see README)." >&2
fi
