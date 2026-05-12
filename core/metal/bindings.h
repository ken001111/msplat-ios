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
// Populated when MSPLAT_2DGS=1 routes through nd_rasterize_forward_2dgs_kernel;
// undefined / zero otherwise. Phase 2b.3.2 (6/N) smoke test reads these to
// validate the dispatch path end-to-end.
MTensor msplat_last_out_depth();    // (H, W) Float32 — alpha-weighted depth
MTensor msplat_last_out_normal();   // (H, W, 3) Float32 — alpha-weighted world-space normal

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
