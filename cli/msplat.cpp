#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <CLI/CLI.hpp>
#include "model.hpp"
#include "input_data.hpp"
#include "random_iter.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include "bindings.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    CLI::App app{"msplat — 3D Gaussian Splatting for Apple Silicon"};
    app.set_version_flag("--version", APP_VERSION);

    // Required
    std::string projectRoot;
    app.add_option("input", projectRoot, "Path to dataset (COLMAP, Nerfstudio, Polycam)")
        ->required()
        ->check(CLI::ExistingDirectory);

    // Output
    std::string outputScene = "splat.ply";
    app.add_option("-o,--output", outputScene, "Output scene path");
    int saveEvery = -1;
    app.add_option("-s,--save-every", saveEvery, "Save every N steps (-1 to disable)");

    // Resume
    std::string resume;
    app.add_option("--resume", resume, "Resume training from PLY file")
        ->check(CLI::ExistingFile);

    // Validation
    bool validate = false;
    app.add_flag("--val", validate, "Withhold a camera for validation");
    std::string valImage = "random";
    app.add_option("--val-image", valImage, "Validation image filename");
    std::string valRender;
    app.add_option("--val-render", valRender, "Directory to render validation images");

    // Evaluation
    bool evalMode = false;
    app.add_flag("--eval", evalMode, "Evaluate on held-out test views");
    int testEvery = 8;
    app.add_option("--test-every", testEvery, "Hold out every Nth image for eval")
        ->check(CLI::Range(2, 100));

    // Training hyperparameters
    int numIters = 30000;
    app.add_option("-n,--num-iters", numIters, "Number of iterations")
        ->check(CLI::Range(1, 1000000));
    float downScaleFactor = 1.0f;
    app.add_option("-d,--downscale-factor", downScaleFactor, "Image downscale factor")
        ->check(CLI::Range(1.0f, 32.0f));
    int numDownscales = 2;
    app.add_option("--num-downscales", numDownscales, "Progressive downscale levels");
    int resolutionSchedule = 3000;
    app.add_option("--resolution-schedule", resolutionSchedule, "Double resolution every N steps");
    int shDegree = 3;
    app.add_option("--sh-degree", shDegree, "Max spherical harmonics degree")
        ->check(CLI::Range(0, 4));
    int shDegreeInterval = 1000;
    app.add_option("--sh-degree-interval", shDegreeInterval, "Increase SH degree every N steps");
    float ssimWeight = 0.2f;
    app.add_option("--ssim-weight", ssimWeight, "SSIM loss weight (0 = L1 only)")
        ->check(CLI::Range(0.0f, 1.0f));
    int refineEvery = 100;
    app.add_option("--refine-every", refineEvery, "Densify/prune every N steps");
    int warmupLength = 500;
    app.add_option("--warmup-length", warmupLength, "Steps before first densification");
    int resetAlphaEvery = 30;
    app.add_option("--reset-alpha-every", resetAlphaEvery, "Reset opacity every N refinements");
    float densifyGradThresh = 0.0002f;
    app.add_option("--densify-grad-thresh", densifyGradThresh, "Gradient threshold for split/dup");
    float densifySizeThresh = 0.01f;
    app.add_option("--densify-size-thresh", densifySizeThresh, "Size threshold (dup vs split)");
    int stopScreenSizeAt = 4000;
    app.add_option("--stop-screen-size-at", stopScreenSizeAt, "Stop splitting large gaussians after N steps");
    float splitScreenSize = 0.05f;
    app.add_option("--split-screen-size", splitScreenSize, "Screen-space split threshold");
    bool keepCrs = false;
    app.add_flag("--keep-crs", keepCrs, "Retain input coordinate reference system");
    std::vector<float> bgColor = {0.6130f, 0.0101f, 0.3984f};
    app.add_option("--bg-color", bgColor, "Background RGB (0-1), default magenta")
        ->expected(3);
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");

    // Facescan / Colab parity: per-frame foreground masks.
    std::string masksDir;
    app.add_option("--masks-dir", masksDir,
        "Directory of per-frame foreground masks (PNG, white=foreground)");
    int maskDilation = 8;
    app.add_option("--mask-dilation", maskDilation,
        "Dilate the foreground region by N pixels (matches PIL MaxFilter(2*r+1))");

    // Phase 2b.2a validation hook: dump the initial scales tensor (after Model
    // construction, before any training step) to a binary file, then exit.
    std::string dumpInitScales;
    app.add_option("--dump-init-scales", dumpInitScales,
        "Dump initial scales tensor to <path> and exit (Phase 2b.2a validation)");

    // Phase 2c.0: unproject rendered depth across all training cameras into
    // a world-space point cloud and save as PLY. Requires a loaded scene
    // (either freshly trained or via --resume).
    std::string exportPointCloud;
    app.add_option("--export-point-cloud", exportPointCloud,
        "Unproject depth from all training views, save world-space point cloud PLY at <path>");
    float pointCloudAlphaThresh = 0.5f;
    app.add_option("--point-cloud-alpha-thresh", pointCloudAlphaThresh,
        "Min accumulated alpha for a pixel to contribute a point (default 0.5)");

    // Phase 2c.1: voxel grid sizing for TSDF fusion. --mesh-info just dumps the
    // computed grid params (origin, dims, memory footprint) without doing
    // fusion yet — useful for verifying the sizing is reasonable before 2c.2
    // wires the integration kernel.
    bool meshInfo = false;
    app.add_flag("--mesh-info", meshInfo,
        "Compute + print voxel-grid sizing from camera extents (Phase 2c.1)");
    float meshVoxelSize = 0.004f;   // 4 mm — iPad-friendly default
    app.add_option("--mesh-voxel-size", meshVoxelSize,
        "Voxel size in meters (default 0.004 = 4 mm)");
    float meshBound = 0.3f;          // 30 cm radius — face / small object scale
    app.add_option("--mesh-bound", meshBound,
        "Half-extent of the voxel grid around the camera centroid, meters (default 0.3)");

    // Phase 2c.2: TSDF fusion. Produces a raw [Dz, Dy, Dx, 2] Float32 dump
    // (sdf + weight per voxel) — opens in tools like ImageJ / 3D Slicer with
    // the dims advertised in the --mesh-info banner. Marching Cubes in 2c.3
    // will consume the in-memory grid directly without round-tripping.
    std::string exportTSDF;
    app.add_option("--export-tsdf", exportTSDF,
        "Run TSDF fusion across all training views, save raw grid (Phase 2c.2)");
    float meshAlphaThresh = 0.5f;
    app.add_option("--mesh-alpha-thresh", meshAlphaThresh,
        "Min accumulated alpha for a pixel to contribute (default 0.5)");
    float meshTruncMultiplier = 4.0f;
    app.add_option("--mesh-trunc-voxels", meshTruncMultiplier,
        "TSDF truncation distance in voxel widths (default 4)");

    // Phase 2b.3.2 (6/N) smoke hook: load a PLY (3DGS or 2DGS) and render the
    // first training camera with msplat_render, then exit. Reports min/max/mean
    // and nonzero counts on out_img + the 2DGS side outputs (out_depth /
    // out_normal). Run with MSPLAT_2DGS=1 to exercise the 2DGS forward path.
    std::string renderOnly;
    app.add_option("--render-only", renderOnly,
        "Load <PLY>, render the first training camera, dump stats, then exit");

    CLI11_PARSE(app, argc, argv);

    if (validate || !valRender.empty()) validate = true;
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    downScaleFactor = std::max(downScaleFactor, 1.0f);

    try {
        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);

        for (auto &cam : inputData.cameras)
            cam.loadImage(downScaleFactor);

        if (!masksDir.empty()) {
            applyDepthMasks(inputData, masksDir, maskDilation);
        }

        std::vector<Camera> cams;
        std::vector<Camera> testCams;
        Camera *valCam = nullptr;

        if (evalMode) {
            auto [train, test] = inputData.splitTrainTest(testEvery);
            cams = train; testCams = test;
            std::cout << "Eval mode: " << cams.size() << " train, " << testCams.size() << " test" << std::endl;
        } else {
            auto [train, val] = inputData.getCameras(validate, valImage);
            cams = train; valCam = val;
        }

        Model model(inputData, cams.size(),
                     numDownscales, resolutionSchedule, shDegree, shDegreeInterval,
                     refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh,
                     densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                     numIters, keepCrs,
                     bgColor.data());

        if (!dumpInitScales.empty()) {
            msplat_gpu_sync();
            MTensor cpu = model.scales.cpu();
            std::ofstream o(dumpInitScales, std::ios::binary);
            o.write(reinterpret_cast<const char*>(cpu.data<float>()), cpu.nbytes());
            std::cout << "Dumped " << cpu.numel() << " floats ("
                      << cpu.nbytes() << " bytes) of initial scales to "
                      << dumpInitScales << std::endl;
            return 0;
        }

        if (!renderOnly.empty()) {
            if (cams.empty()) throw std::runtime_error("--render-only: no training cameras available");
            std::cout << "Loading PLY: " << renderOnly << std::endl;
            model.loadPly(renderOnly);
            std::cout << "Loaded " << model.means.size(0) << " gaussians, scales shape ["
                      << model.scales.size(0) << ", " << model.scales.size(1) << "]" << std::endl;

            Camera &cam = cams[0];
            std::cout << "Rendering camera: " << cam.filePath << "  ("
                      << cam.width << "x" << cam.height << ")  fx=" << cam.fx << " fy=" << cam.fy << std::endl;

            MTensor out_img = model.render(cam, 0);
            msplat_gpu_sync();

            auto report = [](const char *name, const MTensor &t) {
                if (!t.defined()) { std::cout << "  " << name << ": <undefined>" << std::endl; return; }
                const float *p = t.data<float>();
                int64_t n = t.numel();
                float mn = p[0], mx = p[0];
                double sum = 0.0;
                int64_t nz = 0;
                for (int64_t i = 0; i < n; i++) {
                    mn = std::min(mn, p[i]); mx = std::max(mx, p[i]); sum += p[i];
                    if (p[i] != 0.0f) nz++;
                }
                std::cout << "  " << std::left << std::setw(11) << name << " shape=[";
                for (size_t k = 0; k < t.shape().size(); k++) {
                    std::cout << t.shape()[k] << (k + 1 < t.shape().size() ? ", " : "");
                }
                std::cout << "]  min=" << std::setw(11) << mn
                          << "  max=" << std::setw(11) << mx
                          << "  mean=" << std::setw(11) << (sum / std::max((int64_t)1, n))
                          << "  nonzero=" << nz << "/" << n << std::endl;
            };
            std::cout << "Render done. Buffer stats:" << std::endl;
            report("out_img",          out_img);
            report("out_depth",        msplat_last_out_depth());
            report("out_normal",       msplat_last_out_normal());
            report("out_alpha",        msplat_last_out_alpha());
            report("out_median_depth", msplat_last_out_median_depth());
            report("out_distortion",   msplat_last_out_distortion());
            return 0;
        }

        // M2.6: 2DGS training loop restored. The deleted 3DGS BENCHMARK
        // harness + --val-render + --save-every will be re-added if needed.
        std::vector<size_t> camIndices(cams.size());
        std::iota(camIndices.begin(), camIndices.end(), 0);
        InfiniteRandomIterator<size_t> camsIter(camIndices);

        size_t step = 1;
        if (!resume.empty()) step = model.loadPly(resume) + 1;

        auto train_t0 = std::chrono::high_resolution_clock::now();
        for (; step <= (size_t)numIters; step++) {
            Camera &cam = cams[camsIter.next()];
            MTensor gt = cam.getGPUImage(model.getDownscaleFactor(step));
            float loss = model.fullIteration(cam, step, gt, ssimWeight);
            model.schedulersStep(step);
            model.afterTrain(step);
            msplat_commit();

            if (step == 1 || step % 100 == 0 || step == (size_t)numIters) {
                msplat_gpu_sync();
                std::cout << "  iter " << std::setw(6) << step
                          << "  loss=" << std::fixed << std::setprecision(6) << loss
                          << "  N=" << model.means.size(0) << std::endl;
            }
        }
        auto train_t1 = std::chrono::high_resolution_clock::now();
        double train_s = std::chrono::duration_cast<std::chrono::milliseconds>(train_t1 - train_t0).count() / 1000.0;
        std::cout << "Training done in " << train_s << "s (" << numIters << " iters)" << std::endl;

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene, numIters);

        // Phase 2c.0: depth → point cloud export
        if (!exportPointCloud.empty()) {
            auto pc_t0 = std::chrono::high_resolution_clock::now();
            int64_t n = model.exportPointCloud(cams, (int)numIters,
                                               pointCloudAlphaThresh, exportPointCloud);
            auto pc_t1 = std::chrono::high_resolution_clock::now();
            double pc_s = std::chrono::duration_cast<std::chrono::milliseconds>(pc_t1 - pc_t0).count() / 1000.0;
            std::cout << "Exported " << n << " points to " << exportPointCloud
                      << " in " << pc_s << "s" << std::endl;
        }

        // Phase 2c.1: voxel grid sizing (no fusion yet — that's 2c.2).
        if (meshInfo || !exportTSDF.empty()) {
            auto grid = model.makeVoxelGrid(cams, meshVoxelSize, meshBound);
            int64_t numVoxels = (int64_t)grid.dims[0] * grid.dims[1] * grid.dims[2];
            std::cout << "\n=== Voxel grid (Phase 2c.1) ===" << std::endl;
            std::cout << "  origin:      ("
                      << grid.origin[0] << ", " << grid.origin[1] << ", " << grid.origin[2] << ")" << std::endl;
            std::cout << "  dims:        " << grid.dims[0] << " x " << grid.dims[1] << " x " << grid.dims[2]
                      << " = " << numVoxels << " voxels" << std::endl;
            std::cout << "  voxel size:  " << grid.voxelSize << " m" << std::endl;
            std::cout << "  extent:      "
                      << (grid.dims[0] * grid.voxelSize) << " x "
                      << (grid.dims[1] * grid.voxelSize) << " x "
                      << (grid.dims[2] * grid.voxelSize) << " m" << std::endl;
            std::cout << "  buffer:      " << (numVoxels * 2 * 4) / (1024 * 1024)
                      << " MB (Float32 sdf + weight)" << std::endl;

            // Phase 2c.2: TSDF fusion + raw dump.
            if (!exportTSDF.empty()) {
                float truncDist = meshTruncMultiplier * meshVoxelSize;
                std::cout << "Fusing " << cams.size() << " views (trunc=" << truncDist
                          << " m, alphaThresh=" << meshAlphaThresh << ")..." << std::endl;
                auto t0 = std::chrono::high_resolution_clock::now();
                model.integrateTSDF(grid, cams, (int)numIters, truncDist, meshAlphaThresh);
                auto t1 = std::chrono::high_resolution_clock::now();
                double s = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;
                std::cout << "Fusion done in " << s << "s" << std::endl;

                // Read grid back to CPU + write raw dump.
                msplat_gpu_sync();
                MTensor cpu = grid.data.cpu();
                std::ofstream f(exportTSDF, std::ios::binary);
                f.write(reinterpret_cast<const char*>(cpu.data<float>()), cpu.nbytes());
                std::cout << "Wrote " << cpu.nbytes() / (1024 * 1024) << " MB raw grid to "
                          << exportTSDF << " (layout: [Dz, Dy, Dx, 2] Float32, (sdf, weight))" << std::endl;

                // Diagnostic: count how many voxels got at least one observation.
                const float *p = cpu.data<float>();
                int64_t observed = 0, sign_change = 0;
                for (int64_t i = 0; i < numVoxels; i++) {
                    float w = p[i*2 + 1];
                    if (w > 0) observed++;
                }
                std::cout << "Voxels observed (weight > 0): " << observed << " / " << numVoxels
                          << " (" << (100.0 * observed / numVoxels) << "%)" << std::endl;
            }
        }

        // Evaluation
        if (evalMode && !testCams.empty()) {
            double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
            int nTest = testCams.size();

            std::cout << "\n=== Evaluation (" << nTest << " test views) ===" << std::endl;
            for (int i = 0; i < nTest; i++) {
                MTensor rgb = model.render(testCams[i], numIters);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                MTensor gt_cpu = testCams[i].getGPUImage(model.getDownscaleFactor(numIters)).cpu();

                float p = psnr(rgb_cpu, gt_cpu);
                float s = ssim_eval(rgb_cpu, gt_cpu);
                float l = l1_loss(rgb_cpu, gt_cpu);
                sumPsnr += p; sumSsim += s; sumL1 += l;

                std::cout << "  [" << (i+1) << "/" << nTest << "] "
                          << fs::path(testCams[i].filePath).filename().string()
                          << "  PSNR=" << p << "  SSIM=" << s << "  L1=" << l << std::endl;
            }
            std::cout << "\n  PSNR:  " << (sumPsnr / nTest)
                      << "  SSIM:  " << (sumSsim / nTest)
                      << "  L1:  " << (sumL1 / nTest)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        // Validation
        if (valCam) {
            MTensor rgb = model.render(*valCam, numIters);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = valCam->getGPUImage(model.getDownscaleFactor(numIters)).cpu();

            std::cout << "\n=== Validation (" << valCam->filePath << ") ===" << std::endl;
            std::cout << "  PSNR:  " << psnr(rgb_cpu, gt_cpu)
                      << "  SSIM:  " << ssim_eval(rgb_cpu, gt_cpu)
                      << "  L1:  " << l1_loss(rgb_cpu, gt_cpu)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        cleanup_msplat_metal();
        msplat_gpu_sync();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        cleanup_msplat_metal();
        msplat_gpu_sync();
        return 1;
    }
}
