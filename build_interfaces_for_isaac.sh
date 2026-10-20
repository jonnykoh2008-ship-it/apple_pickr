#!/usr/bin/env bash
# Build appleproj_interfaces for the Python that Isaac Sim runs.
#
# vision_apple_pick.py runs inside Isaac Sim, which ships Python 3.11, while the
# system ROS 2 Jazzy build targets Python 3.12. The 3.12 build cannot be
# imported there, so the script refuses to start without
# APPLEPROJ_INTERFACES_PREFIX pointing at a 3.11 build.
#
# A ROS 2 interface package splits into two kinds of artefact:
#
#   * lib<pkg>__rosidl_*.so   plain C, no Python in them at all
#   * <pkg>_s__rosidl_*.so    the C extension that bridges C to a specific
#                             CPython ABI, and the only version-specific part
#
# So the C libraries are reused from the existing colcon build and only the
# three extension modules are recompiled against Isaac's Python 3.11 headers.
# That avoids standing up a whole second ROS 2 toolchain inside Isaac.
#
# Usage:
#   ./build_interfaces_for_isaac.sh
#   export APPLEPROJ_INTERFACES_PREFIX=<printed path>

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG=appleproj_interfaces
SRC_INSTALL="${PROJECT_DIR}/install/${PKG}"
GEN_DIR="${PROJECT_DIR}/build/${PKG}/rosidl_generator_py/${PKG}"
ISAAC_ROOT=/home/rokey/isaacsim
ISAAC_PY_INC="${ISAAC_ROOT}/kit/python/include/python3.11"
BRIDGE_ROOT="${ISAAC_ROOT}/exts/isaacsim.ros2.bridge/jazzy"
OUT="${PROJECT_DIR}/install_isaac311/${PKG}"
SITE="${OUT}/lib/python3.11/site-packages/${PKG}"

for path in "${SRC_INSTALL}" "${GEN_DIR}" "${ISAAC_PY_INC}" "${BRIDGE_ROOT}/lib"; do
  if [ ! -d "${path}" ]; then
    echo "missing: ${path}" >&2
    if [ "${path}" = "${SRC_INSTALL}" ] || [ "${path}" = "${GEN_DIR}" ]; then
      echo "run 'colcon build --packages-select ${PKG}' first" >&2
    fi
    exit 1
  fi
done

echo "==> preparing ${OUT}"
rm -rf "${OUT}"
mkdir -p "${SITE}" "${OUT}/lib"

# 1. Python-independent C libraries, reused as they are.
echo "==> copying C libraries"
cp -a "${SRC_INSTALL}/lib/"lib${PKG}__*.so "${OUT}/lib/"
COPIED_LIBS=$(ls "${OUT}/lib" | wc -l)

# 2. Pure Python modules. The .pyc files are 3.12 bytecode, so leave them out
#    and let 3.11 compile its own on first import.
echo "==> copying Python modules"
SRC_SITE="${SRC_INSTALL}/lib/python3.12/site-packages/${PKG}"
(cd "${SRC_SITE}" && find . -name '*.py' -print0 | tar --null -cf - -T -) \
  | (cd "${SITE}" && tar -xf -)

# 3. The version-specific extensions, rebuilt for 3.11.
echo "==> compiling extensions against Python 3.11"
# The generated sources include numpy headers, and the extension is loaded by
# Isaac's interpreter, so it has to be built against the numpy Isaac itself
# ships. Building against the system numpy would compile fine and then break at
# runtime on an ABI mismatch.
NUMPY_INC="$("${ISAAC_ROOT}/python.sh" -c 'import numpy; print(numpy.get_include())' 2>/dev/null | tail -1)"
if [ ! -d "${NUMPY_INC}" ]; then
  echo "could not locate Isaac's numpy headers (got '${NUMPY_INC}')" >&2
  exit 1
fi
echo "==> numpy headers: ${NUMPY_INC}"

INCLUDES=(
  "-I${ISAAC_PY_INC}"
  "-I${NUMPY_INC}"
  "-I${SRC_INSTALL}/include"
  "-I${SRC_INSTALL}/include/${PKG}"
)
for dir in /opt/ros/jazzy/include/*/; do
  INCLUDES+=("-I${dir%/}")
done
INCLUDES+=("-I/opt/ros/jazzy/include")

BUILT=0
for typesupport in rosidl_typesupport_c rosidl_typesupport_fastrtps_c \
                   rosidl_typesupport_introspection_c; do
  entry="${GEN_DIR}/_${PKG}_s.ep.${typesupport}.c"
  if [ ! -f "${entry}" ]; then
    echo "   skip ${typesupport}: no generated entry point" >&2
    continue
  fi
  sources=("${entry}")
  while IFS= read -r -d '' file; do
    sources+=("${file}")
  done < <(find "${GEN_DIR}" -name '*_s.c' -print0)

  # The generated code calls convert_to_py/convert_from_py of every package the
  # messages embed -- builtin_interfaces for Header stamps, std_msgs, and so on.
  # Those symbols live in each dependency's rosidl_generator_py library, which
  # Isaac already ships, so link them rather than leaving the extension with
  # undefined symbols that only surface as a segfault at import time.
  deps=()
  for dep in builtin_interfaces std_msgs geometry_msgs sensor_msgs \
             action_msgs unique_identifier_msgs; do
    if [ -f "${BRIDGE_ROOT}/lib/lib${dep}__rosidl_generator_py.so" ]; then
      deps+=("-l${dep}__rosidl_generator_py" "-l${dep}__${typesupport}")
    fi
  done

  target="${SITE}/${PKG}_s__${typesupport}.so"
  gcc -shared -fPIC -O2 -o "${target}" "${sources[@]}" \
    "${INCLUDES[@]}" \
    -L"${OUT}/lib" -L"${BRIDGE_ROOT}/lib" \
    -l${PKG}__${typesupport} -l${PKG}__rosidl_generator_c \
    "${deps[@]}" \
    -Wl,-rpath,"${OUT}/lib" -Wl,-rpath,"${BRIDGE_ROOT}/lib" \
    2> >(grep -v 'warning:' >&2) || {
      echo "   FAILED ${typesupport}" >&2
      exit 1
    }
  # An undefined symbol here becomes a segfault when Isaac imports the module,
  # so fail now, where the message is readable.
  if undefined=$(ldd -r "${target}" 2>&1 | grep 'undefined symbol' | head -3) \
     && [ -n "${undefined}" ]; then
    echo "   FAILED ${typesupport}: unresolved symbols" >&2
    echo "${undefined}" >&2
    exit 1
  fi
  echo "   built ${PKG}_s__${typesupport}.so"
  BUILT=$((BUILT + 1))
done

echo
echo "libraries : ${COPIED_LIBS}"
echo "extensions: ${BUILT}"
echo "prefix    : ${OUT}"
echo
echo "export APPLEPROJ_INTERFACES_PREFIX=${OUT}"
