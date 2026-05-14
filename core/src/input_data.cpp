#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading ───────────────────────────────────────────────────────────

void Camera::loadImage(float downscaleFactor) {
    Image raw = imreadRGB(filePath);
    if (raw.empty()) return;

    // If actual image dimensions differ from metadata, rescale intrinsics
    if (width > 0 && height > 0 && (raw.width != width || raw.height != height)) {
        float sx = (float)raw.width / (float)width;
        float sy = (float)raw.height / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = raw.width; height = raw.height;
    } else if (width == 0 || height == 0) {
        width = raw.width; height = raw.height;
    }

    // Downscale
    if (downscaleFactor > 1.0f) {
        int newW = (int)(width / downscaleFactor);
        int newH = (int)(height / downscaleFactor);
        raw = resizeArea(raw, newW, newH);
        float s = 1.0f / downscaleFactor;
        fx *= s; fy *= s; cx *= s; cy *= s;
        width = newW; height = newH;
    }

    // Undistort if needed
    if (hasDistortion()) {
        auto result = undistortImage(raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        raw = std::move(result.image);
        fx = result.fx; fy = result.fy;
        cx = result.cx; cy = result.cy;
        width = result.width; height = result.height;
        k1 = k2 = k3 = p1 = p2 = 0;
    }

    image = std::move(raw);
}

Image Camera::getImage(int downscaleFactor) {
    if (downscaleFactor <= 1) return image;

    auto it = imagePyramids.find(downscaleFactor);
    if (it != imagePyramids.end()) return it->second;

    int newW = image.width / downscaleFactor;
    int newH = image.height / downscaleFactor;
    Image scaled = resizeArea(image, newW, newH);
    imagePyramids[downscaleFactor] = scaled;
    return scaled;
}

MTensor& Camera::getGPUImage(int downscaleFactor) {
    auto it = mtensorImageCache.find(downscaleFactor);
    if (it != mtensorImageCache.end()) return it->second;
    Image img = getImage(downscaleFactor);
    MTensor mt = gpu_empty({img.height, img.width, 3}, DType::Float32);
    memcpy(mt.data_ptr(), img.ptr(), img.width * img.height * 3 * sizeof(float));
    mtensorImageCache[downscaleFactor] = mt;
    return mtensorImageCache[downscaleFactor];
}

// ── Scale & center ──────────────────────────────────────────────────────────

// Mirror of Rear_Camera_02_clean.ipynb cell 2 (face-centering + scale-up).
// faceDistance = assumed camera-to-face distance in meters (default 0.4).
// scaleFactor = world scale-up for 2DGS training stability (default 10).
void faceCenteredTransform(InputData &data, float faceDistance, float scaleFactor) {
    if (data.cameras.empty()) return;

    // Use first camera's camToWorld to locate the face in world coords:
    //   face_world = M0 · (0, 0, -faceDistance, 1)  (camera looks down -Z in OpenGL)
    const float *M0 = data.cameras.front().camToWorld;
    float fx_w = M0[2]  * (-faceDistance) + M0[3];
    float fy_w = M0[6]  * (-faceDistance) + M0[7];
    float fz_w = M0[10] * (-faceDistance) + M0[11];

    data.translation[0] = fx_w;
    data.translation[1] = fy_w;
    data.translation[2] = fz_w;
    data.scale = scaleFactor;

    // (cam_pos - face_world) * scaleFactor for every camera.
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  = (cam.camToWorld[3]  - fx_w) * scaleFactor;
        cam.camToWorld[7]  = (cam.camToWorld[7]  - fy_w) * scaleFactor;
        cam.camToWorld[11] = (cam.camToWorld[11] - fz_w) * scaleFactor;
    }

    // Apply to point cloud (no-op when called before random init).
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - fx_w) * scaleFactor;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - fy_w) * scaleFactor;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - fz_w) * scaleFactor;
    }
}

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    float mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += cam.camToWorld[3];   // column 3 of row 0
        mean[1] += cam.camToWorld[7];   // column 3 of row 1
        mean[2] += cam.camToWorld[11];  // column 3 of row 2
    }
    int n = (int)data.cameras.size();
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    data.translation[0] = mean[0];
    data.translation[1] = mean[1];
    data.translation[2] = mean[2];

    // Center camera poses
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  -= mean[0];
        cam.camToWorld[7]  -= mean[1];
        cam.camToWorld[11] -= mean[2];
    }

    // Compute scale from max absolute camera position
    float maxAbs = 0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[3]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[7]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[11]));
    }
    data.scale = (maxAbs > 0) ? (1.0f / maxAbs) : 1.0f;

    // Apply scale to camera positions
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - mean[0]) * data.scale;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - mean[1]) * data.scale;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - mean[2]) * data.scale;
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<Camera>, Camera*> InputData::getCameras(bool validate, const std::string &valImage) {
    if (!validate) return {cameras, nullptr};

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    Camera *valCam = &cameras[valIdx];
    std::vector<Camera> train;
    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back(cameras[i]);

    return {train, valCam};
}

std::tuple<std::vector<Camera>, std::vector<Camera>> InputData::splitTrainTest(int testEvery) {
    std::vector<Camera> train, test;
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back(cameras[i]);
        else
            train.push_back(cameras[i]);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── LiDAR foreground mask application (Facescan / Colab parity) ─────────────

namespace {

// Read a grayscale image by sampling the R channel of an RGB load.
// CoreGraphics expands single-channel PNGs to RGBA when decoding, so R == gray.
std::vector<uint8_t> readMaskGray(const std::string &path, int &w, int &h) {
    Image rgb = imreadRGB(path);
    w = rgb.width; h = rgb.height;
    std::vector<uint8_t> gray((size_t)w * h);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        gray[i] = (uint8_t)(rgb.data[i * 3] * 255.0f + 0.5f);
    }
    return gray;
}

// Nearest-neighbor resize of an 8-bit single-channel image.
std::vector<uint8_t> resizeMaskNN(const std::vector<uint8_t> &src,
                                  int sw, int sh, int dw, int dh) {
    std::vector<uint8_t> dst((size_t)dw * dh);
    for (int y = 0; y < dh; y++) {
        int sy = (int)((int64_t)y * sh / dh);
        for (int x = 0; x < dw; x++) {
            int sx = (int)((int64_t)x * sw / dw);
            dst[(size_t)y * dw + x] = src[(size_t)sy * sw + sx];
        }
    }
    return dst;
}

// Separable max-filter dilation, radius r (kernel size 2r+1). Matches the
// behavior of PIL.ImageFilter.MaxFilter(2*r+1) used in Colab.
std::vector<uint8_t> dilateMax(const std::vector<uint8_t> &src, int w, int h, int r) {
    if (r <= 0) return src;
    std::vector<uint8_t> tmp((size_t)w * h);
    for (int y = 0; y < h; y++) {
        const uint8_t *row = &src[(size_t)y * w];
        uint8_t *out = &tmp[(size_t)y * w];
        for (int x = 0; x < w; x++) {
            int x0 = std::max(0, x - r);
            int x1 = std::min(w - 1, x + r);
            uint8_t m = 0;
            for (int xx = x0; xx <= x1; xx++) m = std::max(m, row[xx]);
            out[x] = m;
        }
    }
    std::vector<uint8_t> dst((size_t)w * h);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int y0 = std::max(0, y - r);
            int y1 = std::min(h - 1, y + r);
            uint8_t m = 0;
            for (int yy = y0; yy <= y1; yy++) m = std::max(m, tmp[(size_t)yy * w + x]);
            dst[(size_t)y * w + x] = m;
        }
    }
    return dst;
}

} // anon

void applyDepthMasks(InputData &data, const std::string &masksDir, int dilationPx) {
    fs::path mdir(masksDir);
    if (!fs::exists(mdir) || !fs::is_directory(mdir)) {
        throw std::runtime_error("Masks directory not found: " + masksDir);
    }

    int applied = 0, missing = 0;
    for (auto &cam : data.cameras) {
        if (cam.image.empty()) continue;

        fs::path imgPath(cam.filePath);
        fs::path maskPath = mdir / (imgPath.stem().string() + ".png");
        if (!fs::exists(maskPath)) { missing++; continue; }

        int mw, mh;
        std::vector<uint8_t> mask = readMaskGray(maskPath.string(), mw, mh);
        if (mw != cam.image.width || mh != cam.image.height) {
            mask = resizeMaskNN(mask, mw, mh, cam.image.width, cam.image.height);
        }
        if (dilationPx > 0) {
            mask = dilateMax(mask, cam.image.width, cam.image.height, dilationPx);
        }

        // Background → white. Matches Colab's `img[mask < 128] = [255,255,255]`.
        const size_t N = (size_t)cam.image.width * cam.image.height;
        for (size_t i = 0; i < N; i++) {
            if (mask[i] < 128) {
                cam.image.data[i * 3 + 0] = 1.0f;
                cam.image.data[i * 3 + 1] = 1.0f;
                cam.image.data[i * 3 + 2] = 1.0f;
            }
        }
        applied++;
    }
    std::cerr << "msplat: applied " << applied << " depth masks (dilation=" << dilationPx
              << "px, missing=" << missing << ")\n";
}

// ── Random point cloud init (fallback when no SfM points present) ───────────

void initializeRandomPoints(InputData &data, int64_t numPoints, float extent) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distXyz(-extent, extent);
    std::uniform_int_distribution<int> distRgb(0, 255);

    data.points.xyz.resize(numPoints * 3);
    data.points.rgb.resize(numPoints * 3);
    for (int64_t i = 0; i < numPoints; i++) {
        data.points.xyz[i*3+0] = distXyz(rng);
        data.points.xyz[i*3+1] = distXyz(rng);
        data.points.xyz[i*3+2] = distXyz(rng);
        data.points.rgb[i*3+0] = (uint8_t)distRgb(rng);
        data.points.rgb[i*3+1] = (uint8_t)distRgb(rng);
        data.points.rgb[i*3+2] = (uint8_t)distRgb(rng);
    }
    data.points.count = numPoints;
}

void initializeRandomPointsCameraAware(InputData &data, int64_t numPoints, float radiusFrac) {
    // Bail to dumb init if there are no cameras to derive a bbox from.
    if (data.cameras.empty()) {
        initializeRandomPoints(data, numPoints, 1.3f);
        return;
    }

    // Scene center is the WORLD ORIGIN regardless of camera distribution.
    // Both face-centered preprocessing and autoScaleAndCenter put the scene
    // at (0, 0, 0). Earlier this function used the *camera centroid* which
    // for partial-orbit captures (like the dummyhead, where cameras only cover
    // a forward-facing arc) is offset from the actual scene by 1-3 world
    // units — every gaussian then starts on the wrong side of the cameras and
    // training cannot recover the geometry in 5000 iters.
    float center[3] = {0.0f, 0.0f, 0.0f};
    float invN = 1.0f / (float)data.cameras.size();

    // Scene radius proxy = mean camera-to-origin distance × radiusFrac. For
    // face-centered preprocessing this is approximately scaleFactor·faceDistance·radiusFrac
    // (= 10·0.4·0.35 ≈ 1.4), which neatly covers a ~14 cm scaled-face region.
    float meanDist = 0.0f;
    for (const auto &c : data.cameras) {
        float dx = c.camToWorld[3];
        float dy = c.camToWorld[7];
        float dz = c.camToWorld[11];
        meanDist += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    meanDist *= invN;

    float sceneRadius = std::max(meanDist * radiusFrac, 0.05f);

    std::cerr << "msplat: random init from camera bbox — center=("
              << center[0] << "," << center[1] << "," << center[2]
              << ") radius=" << sceneRadius << " points=" << numPoints << "\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> distGauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> distR(0.0f, 1.0f);
    // Match hbb1 reference dataset_readers.py line 241: dark random init
    // (SH coeffs ∈ [0, 1/255] before SH2RGB, ≈ uchar 0-1 after). Forces
    // gaussians to fade in and learn the correct surfaces rather than
    // saturating early with bright random colors that fit "any pixel"
    // (which leads to local minima where pixel loss is low but geometry
    // is wrong).
    std::uniform_int_distribution<int> distRgb(0, 1);

    // Reject-sampling-free spherical: gaussian direction × cuberoot(uniform) radius.
    data.points.xyz.resize(numPoints * 3);
    data.points.rgb.resize(numPoints * 3);
    for (int64_t i = 0; i < numPoints; i++) {
        float gx = distGauss(rng), gy = distGauss(rng), gz = distGauss(rng);
        float invLen = 1.0f / std::sqrt(gx * gx + gy * gy + gz * gz + 1e-12f);
        float r = sceneRadius * std::cbrt(distR(rng));
        data.points.xyz[i*3+0] = center[0] + gx * invLen * r;
        data.points.xyz[i*3+1] = center[1] + gy * invLen * r;
        data.points.xyz[i*3+2] = center[2] + gz * invLen * r;
        data.points.rgb[i*3+0] = (uint8_t)distRgb(rng);
        data.points.rgb[i*3+1] = (uint8_t)distRgb(rng);
        data.points.rgb[i*3+2] = (uint8_t)distRgb(rng);
    }
    data.points.count = numPoints;
}

// ── Format dispatcher ───────────────────────────────────────────────────────

InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath) {
    fs::path root(path);

    InputData data;
    if (fs::exists(root / "transforms.json")) {
        data = loaders::loadNerfstudio(path);
    } else if (fs::exists(root / "cameras.bin") || fs::exists(root / "sparse" / "0" / "cameras.bin")) {
        data = loaders::loadColmap(path, colmapImagePath);
    } else if (fs::exists(root / "keyframes" / "corrected_cameras") || fs::exists(root / "cameras.json")) {
        data = loaders::loadPolycam(path);
    } else {
        throw std::runtime_error("Unrecognized dataset format in: " + path +
            "\nSupported: COLMAP (cameras.bin), Nerfstudio (transforms.json), Polycam (keyframes/)");
    }

    if (data.points.count == 0) {
        std::cerr << "msplat: no SfM point cloud found, seeding camera-aware random init\n";
        initializeRandomPointsCameraAware(data);
    }

    return data;
}
