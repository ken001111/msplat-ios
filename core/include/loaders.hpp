#ifndef LOADERS_H
#define LOADERS_H

#include "input_data.hpp"

// Format-specific loaders
namespace loaders {
    InputData loadColmap(const std::string &projectRoot, const std::string &imageSourcePath = "");
    InputData loadNerfstudio(const std::string &projectRoot);
    InputData loadPolycam(const std::string &projectRoot);
}

// PLY point cloud reader
Points readPly(const std::string &path);

// COLMAP binary point cloud reader
Points readColmapPoints(const std::string &path);

// Image I/O
Image imreadRGB(const std::string &path);       // returns float32 [0,1] directly
bool imreadDimensions(const std::string &path, int &w, int &h);  // pixel dims only, no decode
Image resizeArea(const Image &src, int dstW, int dstH);  // box-filter downscale
void imwriteRGB(const std::string &path, const Image &img);  // save as PNG

// Undistortion (Brown-Conrady model, alpha=0 crop)
struct UndistortResult {
    Image image;
    float fx, fy, cx, cy;  // updated intrinsics after crop
    int width, height;      // cropped dimensions
};
UndistortResult undistortImage(const Image &src,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3);

// Pose utilities
void autoScaleAndCenter(InputData &data);

// Colab face-centered transform: matches notebook cell 2.
// Places the assumed face point (faceDistance meters in front of camera 0,
// along its -Z axis in OpenGL convention) at world origin, then scales the
// world by scaleFactor. This puts the face at a known scale where 2DGS init
// + densify behave consistently (Colab uses faceDistance=0.4, scale=10).
// Run INSTEAD of autoScaleAndCenter, not in addition.
void faceCenteredTransform(InputData &data, float faceDistance, float scaleFactor);

// Gaussian PLY/splat I/O (trained scene export/import)
struct GaussianParams {
    MTensor &means, &scales, &quats, &featuresDc, &featuresRest, &opacities;
    float scale;          // CRS scale factor
    float translation[3]; // CRS translation
    bool keepCrs;
};

void saveGaussianPly(const std::string &path, GaussianParams &p, int step);
void saveGaussianSplat(const std::string &path, GaussianParams &p);

struct LoadedGaussians {
    MTensor means, scales, quats, featuresDc, featuresRest, opacities;
    int step;
};
LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs);

#endif
