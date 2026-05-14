#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Try adding common image extensions if file doesn't exist
static std::string resolveImagePath(const std::string &path) {
    if (fs::exists(path)) return path;
    for (auto ext : {".png", ".jpg", ".jpeg", ".JPG"})
        if (fs::exists(path + ext)) return path + ext;
    return path;
}

InputData loaders::loadNerfstudio(const std::string &projectRoot) {
    std::ifstream f((fs::path(projectRoot) / "transforms.json").string());
    json j = json::parse(f);

    // Global defaults (overridden per-frame if present)
    int gW = j.value("w", 0), gH = j.value("h", 0);
    float gFx = j.value("fl_x", 0.0f), gFy = j.value("fl_y", 0.0f);
    // Older NeRF format ships a single horizontal FOV (camera_angle_x) instead
    // of per-axis focal lengths. We resolve it to fl_x once we know image w.
    bool haveCAX = j.contains("camera_angle_x");
    float cameraAngleX = haveCAX ? j["camera_angle_x"].get<float>() : 0.0f;
    bool haveCAY = j.contains("camera_angle_y");
    float cameraAngleY = haveCAY ? j["camera_angle_y"].get<float>() : 0.0f;
    float gCx = j.value("cx", 0.0f);  // 0 = "derive from W" sentinel; resolved below
    float gCy = j.value("cy", 0.0f);
    bool haveCx = j.contains("cx");
    bool haveCy = j.contains("cy");
    float gK1 = j.value("k1", 0.0f), gK2 = j.value("k2", 0.0f), gK3 = j.value("k3", 0.0f);
    float gP1 = j.value("p1", 0.0f), gP2 = j.value("p2", 0.0f);

    InputData data;

    for (auto &frame : j["frames"]) {
        Camera cam;
        cam.width  = frame.value("w", gW);
        cam.height = frame.value("h", gH);
        cam.fx = frame.value("fl_x", gFx);
        cam.fy = frame.value("fl_y", gFy);
        cam.cx = frame.contains("cx") ? frame["cx"].get<float>() : (haveCx ? gCx : -1.0f);
        cam.cy = frame.contains("cy") ? frame["cy"].get<float>() : (haveCy ? gCy : -1.0f);
        cam.k1 = frame.value("k1", gK1);     cam.k2 = frame.value("k2", gK2);
        cam.k3 = frame.value("k3", gK3);
        cam.p1 = frame.value("p1", gP1);     cam.p2 = frame.value("p2", gP2);

        std::string fp = frame["file_path"].get<std::string>();
        cam.filePath = (fp[0] == '/' || fp[0] == '.')
            ? resolveImagePath(fp)
            : resolveImagePath((fs::path(projectRoot) / fp).string());

        // transform_matrix is 4x4 c2w (OpenGL convention)
        auto &tm = frame["transform_matrix"];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                cam.camToWorld[r*4+c] = tm[r][c].get<float>();

        data.cameras.push_back(cam);
    }

    std::sort(data.cameras.begin(), data.cameras.end(),
        [](const Camera &a, const Camera &b) { return a.filePath < b.filePath; });

    // If intrinsics aren't fully specified in JSON, derive missing fields from
    // the first image's pixel dimensions + camera_angle_x. This handles the
    // original NeRF / Instant-NGP transforms.json variant where only
    // camera_angle_x is present.
    bool needsDeriveDims = false;
    for (auto &cam : data.cameras) {
        if (cam.width <= 0 || cam.height <= 0 || cam.fx <= 0 || cam.fy <= 0 ||
            cam.cx < 0 || cam.cy < 0) {
            needsDeriveDims = true;
            break;
        }
    }
    if (needsDeriveDims && !data.cameras.empty()) {
        int probeW = 0, probeH = 0;
        if (!imreadDimensions(data.cameras.front().filePath, probeW, probeH)) {
            throw std::runtime_error(
                "Nerfstudio loader: transforms.json is missing intrinsics and the "
                "first image file could not be opened to derive them: " +
                data.cameras.front().filePath);
        }
        std::cerr << "msplat: derived image dims " << probeW << "x" << probeH
                  << " from first frame; resolving missing intrinsics\n";
        for (auto &cam : data.cameras) {
            if (cam.width  <= 0) cam.width  = probeW;
            if (cam.height <= 0) cam.height = probeH;
            // Per-axis focal: prefer fl_x/fl_y; else camera_angle_x/y.
            if (cam.fx <= 0.0f) {
                float ang = haveCAX ? cameraAngleX : (haveCAY ? cameraAngleY : 0.0f);
                if (ang > 0.0f) cam.fx = (float)cam.width  * 0.5f / std::tan(0.5f * ang);
            }
            if (cam.fy <= 0.0f) {
                // Square pixels: fy == fx in pixel units when fl_y absent.
                cam.fy = cam.fx;
                if (haveCAY) cam.fy = (float)cam.height * 0.5f / std::tan(0.5f * cameraAngleY);
            }
            if (cam.cx < 0.0f) cam.cx = (float)cam.width  * 0.5f;
            if (cam.cy < 0.0f) cam.cy = (float)cam.height * 0.5f;
        }
    } else {
        // Even when w/h/fl_x are present, cx/cy may have come through as -1
        // sentinel (no explicit cx in JSON, no global default). Resolve to
        // image center.
        for (auto &cam : data.cameras) {
            if (cam.cx < 0.0f) cam.cx = (float)cam.width  * 0.5f;
            if (cam.cy < 0.0f) cam.cy = (float)cam.height * 0.5f;
        }
    }

    // Point cloud
    if (j.contains("ply_file_path")) {
        std::string p = j["ply_file_path"].get<std::string>();
        if (p[0] != '/') p = (fs::path(projectRoot) / p).string();
        if (fs::exists(p)) data.points = readPly(p);
    }
    if (data.points.count == 0) {
        // Accept both casings — hbb1 2DGS reference uses lowercase points3d.ply,
        // COLMAP / our iOS-app captures use uppercase points3D.ply.
        for (auto p : {"sparse/0/points3D.ply", "points3D.ply", "points3d.ply"}) {
            auto path = (fs::path(projectRoot) / p).string();
            if (fs::exists(path)) { data.points = readPly(path); break; }
        }
    }

    // Face-centered Colab-style transform when MSPLAT_FACE_DISTANCE env var is
    // set (e.g. 0.4). MSPLAT_SCALE_FACTOR defaults to 10. Both are optional —
    // unset → fall back to the generic autoScaleAndCenter (orbit-cap normalization).
    const char *envFaceDist = std::getenv("MSPLAT_FACE_DISTANCE");
    if (envFaceDist && std::atof(envFaceDist) > 0.0f) {
        float faceDist = (float)std::atof(envFaceDist);
        const char *envScale = std::getenv("MSPLAT_SCALE_FACTOR");
        float scaleF = envScale ? (float)std::atof(envScale) : 10.0f;
        std::cerr << "msplat: Colab-style face-centered transform (faceDistance="
                  << faceDist << "m, scaleFactor=" << scaleF << ")\n";
        faceCenteredTransform(data, faceDist, scaleF);
    } else {
        autoScaleAndCenter(data);
    }
    return data;
}
