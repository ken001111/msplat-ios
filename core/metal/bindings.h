#ifndef MSPLAT_BINDINGS_H
#define MSPLAT_BINDINGS_H

#include <tuple>
#include "metal_tensor.hpp"

// Release all cached GPU tensors (call before exit to prevent GPU memory leak)
void cleanup_msplat_metal();

// Returns the Metal device used by the msplat context (void* in C++, id<MTLDevice> in ObjC++)
#ifdef __OBJC__
id<MTLDevice> msplat_device();
#else
void* msplat_device();
#endif

// GPU tensor allocation (callable from C++ — delegates to Metal device)
MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype);
MTensor gpu_empty(std::vector<int64_t> shape, DType dtype);

// Commit current command buffer (non-blocking)
void msplat_commit();

// Synchronize (commit + wait for completion)
void msplat_gpu_sync();

// GPU timing — non-invasive, uses completion handlers on committed CBs
void msplat_enable_gpu_timing(bool enable);
// Drains accumulated GPU times (ms per CB) into the provided vector. Thread-safe.
void msplat_drain_gpu_times(std::vector<double>& out);
// Drains per-stage GPU times. stage_times must be an array of N_STAGES vectors.
void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names);

// 2DGS forward side outputs from the most recent msplat_render call.
// Populated by nd_rasterize_forward_2dgs_kernel — the only render path now.
MTensor msplat_last_out_depth();          // (H, W) Float32 — alpha-weighted depth
MTensor msplat_last_out_normal();         // (H, W, 3) Float32 — alpha-weighted world-space normal
// M2.1: backward-replay state + regularizer inputs.
MTensor msplat_last_out_alpha();          // (H, W) Float32 — 1 - T_final
MTensor msplat_last_out_median_depth();   // (H, W) Float32 — depth where T crosses 0.5
MTensor msplat_last_out_distortion();     // (H, W) Float32 — depth-distortion regularizer

// M2.5: full 2DGS training step (forward + loss + backward — no Adam yet).
// Encodes the forward 2DGS pipeline, computes L1+distortion loss against gt,
// runs the backward rasterizer + project + SH chain rule. Returns
// (radii [N], loss_value). The per-gaussian gradient buffers live in
// g_tcache and are accessed via msplat_last_dL_d* accessors below.
std::tuple<MTensor, float> msplat_train_step_2dgs(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, int features_rest_bases,
    float lambda_l1, float lambda_dist, float lambda_dssim,
    // Densify gradient stats (accumulated per-iter; host resets after densify):
    MTensor &xys_grad_norm,
    MTensor &vis_counts,
    MTensor &max_2d_size,
    float inv_max_dim);

// Phase 2c.3: Marching Cubes on a TSDF voxel grid. Allocates output buffers
// internally based on `maxTriangles`, dispatches the kernel, reads back the
// emitted triangle count + vertex data. Returns the number of triangles
// written (capped at maxTriangles). Vertex data is xyz triples, 3 per
// triangle, in world space.
int64_t msplat_marching_cubes(
    MTensor &grid,                   // [Dz, Dy, Dx, 2] Float32 sdf+weight
    int Dx, int Dy, int Dz,
    float origin_x, float origin_y, float origin_z,
    float voxelSize,
    int maxTriangles,                // output buffer cap
    std::vector<float> &triangles);  // OUT: (Ntri × 9) world-space vertex coords

// Phase 2c.2: integrate one rendered depth+alpha map into the TSDF voxel grid.
// One Metal thread per voxel; the dispatch is synchronous so multiple calls
// can be chained for multi-camera fusion without atomics.
void msplat_tsdf_integrate(
    MTensor &grid,                  // [Dz, Dy, Dx, 2] Float32 (sdf, weight) in-place
    int Dx, int Dy, int Dz,
    float origin_x, float origin_y, float origin_z,
    float voxelSize,
    MTensor &viewmat,               // 4x4 row-major
    float fx, float fy, float cx, float cy,
    uint32_t imgW, uint32_t imgH,
    float truncDist,
    float alphaThresh,
    MTensor &depthMap,              // [H, W] Float32 (out_depth)
    MTensor &alphaMap);             // [H, W] Float32 (out_alpha)

// M2.6: dispatch fused_adam_kernel on a single parameter group. Sized in
// flat float count; for [N, K] tensors pass n = N*K. Asynchronous — caller
// is expected to msplat_commit + msplat_gpu_sync at end-of-iter.
void msplat_fused_adam(
    MTensor &params, MTensor &grads,
    MTensor &exp_avg, MTensor &exp_avg_sq,
    uint32_t n,
    float step_size, float beta1, float beta2,
    float bc2_sqrt, float eps);

// M2.3/M2.4 per-gaussian gradient accessors — read by Model::fullIteration
// to feed into fused_adam_kernel (M2.6).
MTensor msplat_last_dL_dmean3D();
MTensor msplat_last_dL_dscale();
MTensor msplat_last_dL_dquat();
MTensor msplat_last_dL_dopacity();
MTensor msplat_last_dL_dfeatures_dc();
MTensor msplat_last_dL_dfeatures_rest();
MTensor msplat_last_dL_dmean2D();   // overwritten by project backward for densify stats
MTensor msplat_last_radii();        // forward output (per-gaussian int)
// Diagnostic accessors (intermediate buffers between rasterize_bwd and proj_bwd).
MTensor msplat_last_dL_dcolors();    // [N, 3] — raw SH-output color gradient
MTensor msplat_last_dL_dtransMat();  // [N, 9] — raw transMat gradient (slots 6,7 = depth row x,y)
MTensor msplat_last_dL_dnormal3D();  // [N, 3] — world-space normal gradient
MTensor msplat_last_out_img();       // [H, W, 3] — last rendered image
MTensor msplat_last_dL_dout_img();   // [H, W, 3] — loss-stage gradient on out_img
MTensor msplat_last_final_idx();     // [H, W, 2] int — (last_contributor, median_contributor) per pixel
MTensor msplat_last_final_Ts();      // [H, W, 3] — (T_final, M1, M2) per pixel
MTensor msplat_last_xys();           // [N, 2] — projected screen-space center
MTensor msplat_last_depths();        // [N] — view-space z per gaussian
MTensor msplat_last_num_tiles_hit(); // [N] int — tile area per gaussian

// Render-only forward pass (no loss computation)
// Returns: out_img (H, W, 3) as MTensor
MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
);

// msplat_train_step removed — 3DGS training path retired in Phase 2b cleanup.
// The 2DGS backward port + corresponding train_step land in Milestone 2.

int msplat_densify(
    int N, int buf_capacity,
    float grad_thresh, float size_thresh, float screen_thresh, int check_screen,
    float cull_alpha_thresh, float cull_scale_thresh, float cull_screen_size, int check_huge,
    MTensor &xys_grad_norm, MTensor &vis_counts, MTensor &max_2d_size,
    float half_max_dim,
    MTensor &means_buf, MTensor &scales_buf, MTensor &quats_buf,
    MTensor &featuresDc_buf, MTensor &featuresRest_buf, MTensor &opacities_buf,
    int fr_stride,
    MTensor adam_exp_avg_buf[], MTensor adam_exp_avg_sq_buf[],
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &keep_flag, MTensor &keep_prefix,
    MTensor &block_totals, MTensor &compact_scratch,
    MTensor &random_samples
);

#endif
