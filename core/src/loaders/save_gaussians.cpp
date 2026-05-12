#include "loaders.hpp"
#include "model.hpp"
#include "msplat.hpp"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

static const double C0 = 0.28209479177387814;

void saveGaussianPly(const std::string &path, GaussianParams &p, int step) {
    msplat_gpu_sync();

    std::ofstream o(path, std::ios::binary);
    int64_t N = p.means.size(0);
    int numDc = (int)p.featuresDc.size(1);
    int frBases = (int)p.featuresRest.size(-2);
    int numFr = frBases * 3;
    // 2DGS scenes carry 2 in-plane scales; 3DGS carries 3 axis scales.
    // Driven off the tensor shape so the writer tracks kScaleDim automatically.
    int numScales = (int)p.scales.size(1);

    o << "ply\nformat binary_little_endian 1.0\n";
    o << "comment msplat v" << step << "\n";
    o << "element vertex " << N << "\n";
    o << "property float x\nproperty float y\nproperty float z\n";
    o << "property float nx\nproperty float ny\nproperty float nz\n";
    for (int i = 0; i < numDc; i++) o << "property float f_dc_" << i << "\n";
    for (int i = 0; i < numFr; i++) o << "property float f_rest_" << i << "\n";
    o << "property float opacity\n";
    for (int i = 0; i < numScales; i++) o << "property float scale_" << i << "\n";
    o << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    o << "end_header\n";

    int floatsPerRow = 3 + 3 + numDc + numFr + 1 + numScales + 4;
    std::vector<float> row(floatsPerRow);
    const float *mp = p.means.data<float>(), *sp = p.scales.data<float>(), *qp = p.quats.data<float>();
    const float *dp = p.featuresDc.data<float>(), *op = p.opacities.data<float>();
    const float *frp = p.featuresRest.data<float>();

    for (int64_t i = 0; i < N; i++) {
        int c = 0;
        for (int j = 0; j < 3; j++)
            row[c++] = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];
        row[c++] = 0; row[c++] = 0; row[c++] = 0; // normals
        for (int j = 0; j < numDc; j++) row[c++] = dp[i*numDc+j];
        // Transpose [frBases, 3] → [3, frBases] for PLY convention
        for (int ch = 0; ch < 3; ch++)
            for (int b = 0; b < frBases; b++)
                row[c++] = frp[i*frBases*3 + b*3 + ch];
        row[c++] = op[i];
        for (int j = 0; j < numScales; j++)
            row[c++] = p.keepCrs ? std::log(std::exp(sp[i*numScales+j]) / p.scale) : sp[i*numScales+j];
        for (int j = 0; j < 4; j++) row[c++] = qp[i*4+j];

        o.write(reinterpret_cast<const char*>(row.data()), floatsPerRow * sizeof(float));
    }
}

void saveGaussianSplat(const std::string &path, GaussianParams &p) {
    msplat_gpu_sync();

    std::ofstream o(path, std::ios::binary);
    int64_t N = p.means.size(0);
    int numScales = (int)p.scales.size(1);
    const float *mp = p.means.data<float>(), *sp = p.scales.data<float>(), *qp = p.quats.data<float>();
    const float *dp = p.featuresDc.data<float>(), *op = p.opacities.data<float>();

    // Sort by size/opacity (largest first). For 2DGS the size estimate is just
    // the sum of in-plane radii — surfels are flat discs, no third axis to
    // contribute. SPLAT viewers see a slightly different sort order than the
    // 3DGS equivalent, which is acceptable for a thumbnail-tier export.
    std::vector<float> order(N);
    for (int64_t i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = 0; j < numScales; j++) s += std::exp(sp[i*numScales+j]);
        if (p.keepCrs) s /= p.scale;
        order[i] = s / (1.0f + std::exp(-op[i]));
    }
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return order[a] > order[b]; });

    for (int64_t ii = 0; ii < N; ii++) {
        size_t i = idx[ii];
        float m[3];
        for (int j = 0; j < 3; j++) m[j] = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];
        o.write(reinterpret_cast<const char*>(m), 12);

        // SPLAT format is 3DGS-shaped: always 3 scale floats. For 2DGS surfels
        // we pad the missing axis with a small flat value so the viewer renders
        // them as thin discs instead of degenerate points.
        float sc[3] = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < numScales && j < 3; j++)
            sc[j] = p.keepCrs ? (std::exp(sp[i*numScales+j]) / p.scale) : std::exp(sp[i*numScales+j]);
        if (numScales < 3) sc[2] = 1e-5f;
        o.write(reinterpret_cast<const char*>(sc), 12);

        uint8_t rgb[3];
        for (int j = 0; j < 3; j++) rgb[j] = (uint8_t)std::clamp(((double)dp[i*3+j] * C0 + 0.5) * 255.0, 0.0, 255.0);
        o.write(reinterpret_cast<const char*>(rgb), 3);

        float sig = 1.0f / (1.0f + std::exp(-op[i]));
        uint8_t a = (uint8_t)std::clamp(sig * 255.0f, 0.0f, 255.0f);
        o.write(reinterpret_cast<const char*>(&a), 1);

        uint8_t q[4];
        for (int j = 0; j < 4; j++) q[j] = (uint8_t)std::clamp(qp[i*4+j] * 128.0f + 128.0f, 0.0f, 255.0f);
        o.write(reinterpret_cast<const char*>(q), 4);
    }
}

LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs) {
    msplat_gpu_sync();

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open PLY file: " + path);

    // Parse header
    std::string line;
    int numPoints = 0, step = 0;
    int numDc = 0, numFr = 0;
    // Count scale_N properties — 3 for msplat/3DGS, 2 for Colab-trained 2DGS.
    // Driven off the header so both formats round-trip through the same loader.
    int numScales = 0;

    std::getline(f, line); // "ply"
    if (line.find("ply") == std::string::npos) throw std::runtime_error("Not a PLY file: " + path);
    std::getline(f, line); // "format binary_little_endian 1.0"

    while (std::getline(f, line)) {
        if (line == "end_header") break;

        const std::string iterPrefix = "comment Generated by msplat at iteration ";
        if (line.rfind(iterPrefix, 0) == 0)
            step = std::stoi(line.substr(iterPrefix.length()));

        const std::string vertexPrefix = "element vertex ";
        if (line.rfind(vertexPrefix, 0) == 0)
            numPoints = std::stoi(line.substr(vertexPrefix.length()));

        if (line.rfind("property float f_dc_", 0) == 0) numDc++;
        if (line.rfind("property float f_rest_", 0) == 0) numFr++;
        if (line.rfind("property float scale_", 0) == 0) numScales++;
    }

    if (numPoints == 0) throw std::runtime_error("PLY has no vertices");
    if (numScales != 2 && numScales != 3)
        throw std::runtime_error("PLY: expected 2 (2DGS) or 3 (3DGS) scale properties, got " + std::to_string(numScales));
    int frBases = numFr / 3;

    // Read binary data: xyz(3) + normals(3) + f_dc(numDc) + f_rest(numFr) + opacity(1) + scale(numScales) + rot(4)
    std::vector<float> meansRaw(numPoints * 3);
    std::vector<float> dcRaw(numPoints * numDc);
    std::vector<float> frRaw(numPoints * numFr);
    std::vector<float> opRaw(numPoints);
    std::vector<float> scRaw(numPoints * numScales);
    std::vector<float> qtRaw(numPoints * 4);
    float normals[3];

    for (int i = 0; i < numPoints; i++) {
        f.read(reinterpret_cast<char*>(&meansRaw[i*3]), 12);
        f.read(reinterpret_cast<char*>(normals), 12);
        f.read(reinterpret_cast<char*>(&dcRaw[i*numDc]), numDc * 4);
        f.read(reinterpret_cast<char*>(&frRaw[i*numFr]), numFr * 4);
        f.read(reinterpret_cast<char*>(&opRaw[i]), 4);
        f.read(reinterpret_cast<char*>(&scRaw[i*numScales]), numScales * 4);
        f.read(reinterpret_cast<char*>(&qtRaw[i*4]), 16);
    }

    // CRS transform
    if (keepCrs) {
        for (int i = 0; i < numPoints; i++)
            for (int j = 0; j < 3; j++)
                meansRaw[i*3+j] = (meansRaw[i*3+j] - translation[j]) * scale;
        for (int i = 0; i < numPoints * numScales; i++)
            scRaw[i] = std::log(scale * std::exp(scRaw[i]));
    }

    // Upload to GPU
    LoadedGaussians g;
    g.step = step;
    auto upload = [](std::vector<int64_t> shape, const float *src, size_t bytes) {
        MTensor t = gpu_empty(shape, DType::Float32);
        memcpy(t.data_ptr(), src, bytes);
        return t;
    };
    g.means = upload({(int64_t)numPoints, 3}, meansRaw.data(), meansRaw.size() * 4);
    g.featuresDc = upload({(int64_t)numPoints, (int64_t)numDc}, dcRaw.data(), dcRaw.size() * 4);
    g.opacities = upload({(int64_t)numPoints, 1}, opRaw.data(), opRaw.size() * 4);
    g.scales = upload({(int64_t)numPoints, (int64_t)numScales}, scRaw.data(), scRaw.size() * 4);
    g.quats = upload({(int64_t)numPoints, 4}, qtRaw.data(), qtRaw.size() * 4);

    // Transpose featuresRest: PLY [N, 3, frBases] → internal [N, frBases, 3]
    g.featuresRest = gpu_empty({(int64_t)numPoints, (int64_t)frBases, 3}, DType::Float32);
    float *frOut = g.featuresRest.data<float>();
    for (int i = 0; i < numPoints; i++)
        for (int ch = 0; ch < 3; ch++)
            for (int b = 0; b < frBases; b++)
                frOut[i*frBases*3 + b*3 + ch] = frRaw[i*numFr + ch*frBases + b];

    // numPoints and step available via returned struct
    return g;
}
