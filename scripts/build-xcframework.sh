#!/bin/bash
# Build msplat for macOS, iOS device, and iOS simulator, then bundle into
# a single XCFramework. Run from the msplat repo root.
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Build per-platform static libraries ──────────────────────────────────────

build_platform() {
    local label=$1        # human-readable name
    local sdk=$2          # xcrun -sdk value: macosx | iphoneos | iphonesimulator
    local sysname=$3      # CMake CMAKE_SYSTEM_NAME: Darwin | iOS
    local arches=$4       # space-separated arch list, e.g. "arm64" or "arm64 x86_64"
    local build_dir=$5

    echo "=== Building ${label} (${sdk}, ${arches}) ==="
    rm -rf "${build_dir}"
    cmake -B "${build_dir}" -G Xcode \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME="${sysname}" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=$( [ "${sysname}" = "iOS" ] && echo "17.0" || echo "14.0" ) \
        -DCMAKE_OSX_ARCHITECTURES="${arches// /;}" \
        -DMSPLAT_METAL_SDK="${sdk}" \
        -DCMAKE_OSX_SYSROOT="${sdk}"

    xcodebuild -project "${build_dir}/msplat.xcodeproj" \
        -scheme msplat_core \
        -configuration Release \
        -sdk "${sdk}" \
        ONLY_ACTIVE_ARCH=NO \
        ARCHS="${arches}" \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        build
}

build_platform "macOS"     macosx           Darwin "arm64"        build-macos
build_platform "iOS"       iphoneos         iOS    "arm64"        build-ios
build_platform "iOS Sim"   iphonesimulator  iOS    "arm64 x86_64" build-ios-sim

# Locate the produced static archives.
MACOS_LIB="$(find build-macos     -name libmsplat_core.a | head -1)"
IOS_LIB="$(find build-ios         -name libmsplat_core.a | head -1)"
SIM_LIB="$(find build-ios-sim     -name libmsplat_core.a | head -1)"

MACOS_METALLIB="$(find build-macos     -name default.metallib | head -1)"
IOS_METALLIB="$(find build-ios         -name default.metallib | head -1)"
SIM_METALLIB="$(find build-ios-sim     -name default.metallib | head -1)"

for f in "$MACOS_LIB" "$IOS_LIB" "$SIM_LIB" "$MACOS_METALLIB" "$IOS_METALLIB" "$SIM_METALLIB"; do
    if [ -z "$f" ] || [ ! -f "$f" ]; then
        echo "MISSING BUILD ARTIFACT: $f"
        exit 1
    fi
done

# ── Prepare headers + modulemap ──────────────────────────────────────────────

rm -rf build/xcf-headers
mkdir -p build/xcf-headers
cp core/include/msplat_c_api.h build/xcf-headers/
cat > build/xcf-headers/module.modulemap <<'MAP'
module MsplatCore {
    header "msplat_c_api.h"
    export *
}
MAP

# ── Assemble multi-platform XCFramework ──────────────────────────────────────

rm -rf MsplatCore.xcframework
xcodebuild -create-xcframework \
    -library "${MACOS_LIB}"  -headers build/xcf-headers \
    -library "${IOS_LIB}"    -headers build/xcf-headers \
    -library "${SIM_LIB}"    -headers build/xcf-headers \
    -output MsplatCore.xcframework

# ── Copy per-platform metallibs as Swift package resources ───────────────────
#
# A single Swift package can only ship one default.metallib resource. We bake
# the iOS metallib in (since that's the runtime target); macOS/Sim builds will
# need the trainer's runtime to load the macOS or simulator metallib manually
# via msplat_set_metallib_path() before creating any trainer.

mkdir -p swift/Sources/Msplat/Resources
cp "${IOS_METALLIB}"   swift/Sources/Msplat/Resources/default.metallib
cp "${MACOS_METALLIB}" swift/Sources/Msplat/Resources/default.macos.metallib
cp "${SIM_METALLIB}"   swift/Sources/Msplat/Resources/default.sim.metallib

echo ""
echo "=== Done ==="
echo "  MsplatCore.xcframework  (macOS + iOS + iOS Simulator)"
echo "  swift/Sources/Msplat/Resources/default.metallib       (iOS device runtime)"
echo "  swift/Sources/Msplat/Resources/default.macos.metallib (macOS runtime)"
echo "  swift/Sources/Msplat/Resources/default.sim.metallib   (iOS Simulator runtime)"
