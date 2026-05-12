#import "bindings.h"
#define BLOCK_X 16
#define BLOCK_Y 16

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <chrono>
#import <dlfcn.h>
#import <unordered_map>
#import <functional>
#import <array>
#import <mutex>
#import <mach/mach_time.h>

// GPU profiling infrastructure.
// PROFILE_GPU=1: per-CB total GPU time via completion handlers.
// PROFILE_STAGES=1: per-stage GPU time via Metal timestamp counters + separate encoders.
static bool g_gpu_timing_enabled = false;
static bool g_gpu_timing_checked = false;
static std::mutex g_gpu_timing_mutex;
static std::vector<double> g_gpu_times_ms;

// Per-stage profiling
static bool g_profile_stages = false;
static bool g_profile_stages_checked = false;

// Stage names for training pipeline
static const char* g_train_stage_names[] = {
    "blit_zero", "proj_sh_fwd", "prefix_sort_pack", "rast_fwd",
    "loss_fwd_bwd", "rast_bwd", "proj_sh_bwd_adam", "grad_stats"
};
static constexpr int N_TRAIN_STAGES = 8;

static std::mutex g_stage_timing_mutex;
// Per-stage accumulated times (ms), indexed by stage
static std::vector<double> g_stage_times[N_TRAIN_STAGES];
static int g_stage_report_count = 0;

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    dispatch_queue_t d_queue;

    // Command buffer lifecycle using MPSCommandBuffer for commitAndContinue support.
    MPSCommandBuffer* _currentCB = nil;

    id<MTLCommandBuffer> getCommandBuffer() {
        if (!_currentCB) {
            _currentCB = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            [_currentCB retain];
        }
        return _currentCB;
    }
    void commitCB() {
        if (_currentCB) {
            if (g_gpu_timing_enabled) {
                [_currentCB addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    if (gpu_ms > 0) {
                        std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
                        g_gpu_times_ms.push_back(gpu_ms);
                    }
                }];
            }
            [_currentCB commitAndContinue];
        }
    }
    void syncCB() {
        if (_currentCB) {
            [_currentCB commit];
            [_currentCB waitUntilCompleted];
            [_currentCB release];
            _currentCB = nil;
        }
    }

    // Per-stage GPU timestamp profiling (Metal counter sample buffer)
    id<MTLCounterSampleBuffer> counterSampleBuffer;
    bool counterSamplingAvailable = false;
    double ticksToMs = 0.0;  // conversion factor from GPU ticks to milliseconds

    void initCounterSampling() {
        // Need 2 samples per stage (start + end)
        NSUInteger sampleCount = N_TRAIN_STAGES * 2;

        // Find timestamp counter set
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> cs in device.counterSets) {
            if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestampSet = cs;
                break;
            }
        }
        if (!timestampSet) {
            fprintf(stderr, "PROFILE_STAGES: MTLCommonCounterSetTimestamp not available\n");
            return;
        }

        // Check if stage boundary sampling is supported (guaranteed on Apple Silicon)
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            fprintf(stderr, "PROFILE_STAGES: AtStageBoundary sampling not supported\n");
            return;
        }

        MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
        desc.counterSet = timestampSet;
        desc.sampleCount = sampleCount;
        desc.storageMode = MTLStorageModeShared;
        desc.label = @"msplat stage profiling";

        NSError *error = nil;
        counterSampleBuffer = [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!counterSampleBuffer) {
            fprintf(stderr, "PROFILE_STAGES: Failed to create counter sample buffer: %s\n",
                    error.localizedDescription.UTF8String);
            return;
        }

        // Compute ticks-to-ms conversion (Apple Silicon: mach_absolute_time units)
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        ticksToMs = (double)tb.numer / (double)tb.denom / 1e6;

        counterSamplingAvailable = true;
        fprintf(stderr, "PROFILE_STAGES: GPU timestamp profiling enabled (%lu sample slots)\n",
                (unsigned long)sampleCount);
    }

    // 2DGS forward pipeline (the only render path)
    id<MTLComputePipelineState> project_and_sh_forward_2dgs_kernel_cpso;
    id<MTLComputePipelineState> nd_rasterize_forward_2dgs_kernel_cpso;
    id<MTLComputePipelineState> bitonic_sort_per_tile_2dgs_kernel_cpso;
    // 2DGS backward pipeline (M2.2 / M2.3 — dead code until dispatch lands in M2.5)
    id<MTLComputePipelineState> nd_rasterize_backward_2dgs_kernel_cpso;
    id<MTLComputePipelineState> project_and_sh_backward_2dgs_kernel_cpso;
    // Tile-local sorting (shared with densify)
    id<MTLComputePipelineState> scatter_to_prealloc_bins_kernel_cpso;
    // Prefix sum (shared with densify)
    id<MTLComputePipelineState> prefix_sum_kernel_cpso;
    id<MTLComputePipelineState> block_reduce_kernel_cpso;
    id<MTLComputePipelineState> block_scan_propagate_kernel_cpso;
    // Adam optimizer (M2 reuses)
    id<MTLComputePipelineState> fused_adam_kernel_cpso;
    // GPU densification kernels (kScaleDim-parametric, M2 reuses)
    id<MTLComputePipelineState> densify_classify_kernel_cpso;
    id<MTLComputePipelineState> densify_append_split_kernel_cpso;
    id<MTLComputePipelineState> densify_append_dup_kernel_cpso;
    id<MTLComputePipelineState> densify_cull_classify_kernel_cpso;
    id<MTLComputePipelineState> compact_scatter_kernel_cpso;
    id<MTLComputePipelineState> compact_copy_back_kernel_cpso;
};

// Explicit metallib path (set by Swift/Python wrappers before first use)
static char* g_metallib_path = NULL;

extern "C" void msplat_set_metallib_path(const char* path) {
    free(g_metallib_path);
    g_metallib_path = path ? strdup(path) : NULL;
}

MetalContext* init_msplat_metal_context() {
    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    ctx->device = device;
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->d_queue = dispatch_queue_create("com.msplat.metal", DISPATCH_QUEUE_SERIAL);

    // Find precompiled metallib: explicit path (XCFramework/Python) or auto-discover
    NSError *error = nil;
    id<MTLLibrary> metal_library = nil;

    if (g_metallib_path) {
        // Explicit path (set by XCFramework / Swift wrapper)
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:g_metallib_path]];
        metal_library = [device newLibraryWithURL:url error:&error];
    } else {
        // Auto-discover default.metallib next to this library or the executable
        NSFileManager *fm = [NSFileManager defaultManager];

        // 1. Next to this shared library (Python .so / linked .a)
        Dl_info dl_info;
        if (dladdr((void*)init_msplat_metal_context, &dl_info) && dl_info.dli_fname) {
            NSString *dir = [[NSString stringWithUTF8String:dl_info.dli_fname] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if ([fm fileExistsAtPath:path]) {
                metal_library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
        // 2. Next to the main executable (CLI build)
        if (!metal_library) {
            NSString *dir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if (dir && [fm fileExistsAtPath:path]) {
                metal_library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
    }

    if (!metal_library) {
        fprintf(stderr, "msplat: failed to load metallib: %s\n",
                error ? [[error description] UTF8String] : "default.metallib not found");
        free(ctx);
        return NULL;
    }

    auto load = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [metal_library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "msplat: kernel not found: %s\n", [name UTF8String]);
            return nil;
        }
        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
        [fn release];
        if (error) {
            fprintf(stderr, "msplat: failed to create pipeline for %s: %s\n",
                    [name UTF8String], [[error description] UTF8String]);
        }
        return pso;
    };

    // 2DGS forward pipeline
    ctx->project_and_sh_forward_2dgs_kernel_cpso  = load(@"project_and_sh_forward_2dgs_kernel");
    ctx->nd_rasterize_forward_2dgs_kernel_cpso    = load(@"nd_rasterize_forward_2dgs_kernel");
    ctx->bitonic_sort_per_tile_2dgs_kernel_cpso   = load(@"bitonic_sort_per_tile_2dgs_kernel");
    // 2DGS backward pipeline (M2.2 / M2.3 — dead code until dispatch lands in M2.5).
    ctx->nd_rasterize_backward_2dgs_kernel_cpso   = load(@"nd_rasterize_backward_2dgs_kernel");
    ctx->project_and_sh_backward_2dgs_kernel_cpso = load(@"project_and_sh_backward_2dgs_kernel");
    // Tile-local sorting + prefix sum (shared with densify)
    ctx->scatter_to_prealloc_bins_kernel_cpso     = load(@"scatter_to_prealloc_bins_kernel");
    ctx->prefix_sum_kernel_cpso                   = load(@"prefix_sum_kernel");
    ctx->block_reduce_kernel_cpso                 = load(@"block_reduce_kernel");
    ctx->block_scan_propagate_kernel_cpso         = load(@"block_scan_propagate_kernel");
    // Adam (M2 reuses)
    ctx->fused_adam_kernel_cpso                   = load(@"fused_adam_kernel");
    // GPU densification (kScaleDim-parametric, M2 reuses)
    ctx->densify_classify_kernel_cpso             = load(@"densify_classify_kernel");
    ctx->densify_append_split_kernel_cpso         = load(@"densify_append_split_kernel");
    ctx->densify_append_dup_kernel_cpso           = load(@"densify_append_dup_kernel");
    ctx->densify_cull_classify_kernel_cpso        = load(@"densify_cull_classify_kernel");
    ctx->compact_scatter_kernel_cpso              = load(@"compact_scatter_kernel");
    ctx->compact_copy_back_kernel_cpso            = load(@"compact_copy_back_kernel");

    [metal_library release];

    // Initialize counter sampling if PROFILE_STAGES is set
    ctx->counterSampleBuffer = nil;
    ctx->counterSamplingAvailable = false;
    ctx->ticksToMs = 0.0;
    if (std::getenv("PROFILE_STAGES")) {
        g_profile_stages = true;
        g_profile_stages_checked = true;
        ctx->initCounterSampling();
    }

    return ctx;
}

MetalContext* get_global_context() {
    static MetalContext* ctx = NULL;
    if (ctx == NULL) {
        ctx = init_msplat_metal_context();
    }
    return ctx;
}



#define ENC_SCALAR(encoder, x, i) [encoder setBytes:&x length:sizeof(x) atIndex:i]
#define ENC_ARRAY(encoder, x, i) [encoder setBytes:x length:sizeof(x) atIndex:i]
#define ENC_BUF(encoder, x, i) [encoder setBuffer:x.buffer() offset:0 atIndex:i]

id<MTLDevice> msplat_device() {
    return get_global_context()->device;
}

MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype) {
    return mtensor_zeros(get_global_context()->device, std::move(shape), dtype);
}

MTensor gpu_empty(std::vector<int64_t> shape, DType dtype) {
    return mtensor_empty(get_global_context()->device, std::move(shape), dtype);
}

void msplat_commit() {
    if (!g_gpu_timing_checked) {
        g_gpu_timing_enabled = std::getenv("PROFILE_GPU") != nullptr;
        g_gpu_timing_checked = true;
    }
    get_global_context()->commitCB();
}

void msplat_gpu_sync() {
    get_global_context()->syncCB();
}

void msplat_enable_gpu_timing(bool enable) {
    g_gpu_timing_enabled = enable;
    g_gpu_timing_checked = true;
}

void msplat_drain_gpu_times(std::vector<double>& out) {
    std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
    out = std::move(g_gpu_times_ms);
    g_gpu_times_ms.clear();
}

void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names) {
    std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
    n_stages = std::min(max_stages, N_TRAIN_STAGES);
    for (int i = 0; i < n_stages; i++) {
        stage_times[i] = std::move(g_stage_times[i]);
        g_stage_times[i].clear();
        stage_names[i] = g_train_stage_names[i];
    }
}

#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8

// Cached buffer pool — all intermediate GPU buffers are reused across iterations.
// Sizes only change at densification (every 100 steps); between densifications
// this eliminates all per-iteration GPU allocations.
struct FusedTensorCache {
    int fwd_num_points = 0, capacity = 0, img_height = 0, img_width = 0, num_tiles = 0;

    // 2DGS forward intermediates. per-gaussian, per-sorted-surfel, per-pixel.
    MTensor xys, depths, radii_out, num_tiles_hit, colors, aabb;
    MTensor transMats, normal_opacity;                            // per-gaussian
    MTensor gaussian_ids;
    MTensor packed_xy, packed_transmat, packed_normal_opac, packed_rgb;  // per sorted-surfel
    // Per-pixel: out_img + DEPTH/NORMAL_OFFSET. M2.1 adds the backward-replay
    // state (final_Ts = (T_final, M1, M2), final_idx = (last, median)) plus
    // ALPHA / MIDDEPTH / DISTORTION offsets needed by the regularizer losses.
    MTensor out_img, final_Ts, final_idx, out_depth, out_normal;
    MTensor out_alpha, out_median_depth, out_distortion;

    // M2.2 backward gradient accumulators — atomically scattered by the
    // backward rasterizer, then read by the backward projection (M2.3) to
    // produce gradients w.r.t. means3D / scales / quats / opacities / SH.
    MTensor dL_dtransMat;   // [N, 9]  per-gaussian
    MTensor dL_dmean2D;     // [N, 2]  per-gaussian (screen-space, for densify stats)
    MTensor dL_dnormal3D;   // [N, 3]  per-gaussian
    MTensor dL_dopacity;    // [N, 1]  per-gaussian
    MTensor dL_dcolors;     // [N, 3]  per-gaussian (gradient on raw SH-output color)

    // M2.3 backward projection outputs — final per-gaussian gradients consumed
    // by fused_adam in the eventual train_step (M2.5).
    MTensor dL_dmean3D;        // [N, 3]
    MTensor dL_dscale;         // [N, 2]
    MTensor dL_dquat;          // [N, 4]
    MTensor dL_dfeatures_dc;   // [N, 3]
    MTensor dL_dfeatures_rest; // [N, frBases, 3] — allocated on first dispatch (frBases known)
    int     frBases_for_grad = 0;

    // Tile-local sorting buffers (shared with densify)
    MTensor tile_bins;
    MTensor tile_offsets, tile_scatter_counters;
    MTensor prealloc_bins;  // [num_tiles × MAX_TILE_ELEMS] uint64

    // Multi-threadgroup prefix sum temp buffer
    MTensor block_totals;

    // Intersection overflow detection
    MTensor overflow_flag;
    int64_t capacity_multiplier = 16;

    void ensure_forward(int np, int64_t cap, int ih, int iw, int nt,
                        id<MTLDevice> dev) {
        if (np != fwd_num_points) {
            fwd_num_points = np;
            xys = mtensor_empty(dev, {np, 2}, DType::Float32);
            depths = mtensor_empty(dev, {np}, DType::Float32);
            radii_out = mtensor_empty(dev, {np}, DType::Int32);
            num_tiles_hit = mtensor_empty(dev, {np}, DType::Int32);
            colors = mtensor_empty(dev, {np, 3}, DType::Float32);
            aabb = mtensor_empty(dev, {np, 2}, DType::Float32);
            block_totals = mtensor_empty(dev, {(np + 1023) / 1024}, DType::Int32);
            transMats = mtensor_empty(dev, {np, 9}, DType::Float32);
            normal_opacity = mtensor_empty(dev, {np, 4}, DType::Float32);
            // M2.2 backward accumulators (gradients scattered atomically per gaussian).
            dL_dtransMat = mtensor_empty(dev, {np, 9}, DType::Float32);
            dL_dmean2D = mtensor_empty(dev, {np, 2}, DType::Float32);
            dL_dnormal3D = mtensor_empty(dev, {np, 3}, DType::Float32);
            dL_dopacity = mtensor_empty(dev, {np, 1}, DType::Float32);
            dL_dcolors = mtensor_empty(dev, {np, 3}, DType::Float32);
            // M2.3 final per-gaussian gradients.
            dL_dmean3D = mtensor_empty(dev, {np, 3}, DType::Float32);
            dL_dscale = mtensor_empty(dev, {np, 2}, DType::Float32);
            dL_dquat = mtensor_empty(dev, {np, 4}, DType::Float32);
            dL_dfeatures_dc = mtensor_empty(dev, {np, 3}, DType::Float32);
            // dL_dfeatures_rest allocated lazily — depends on SH degree.
            dL_dfeatures_rest = MTensor{};
            frBases_for_grad = 0;
        }
        if (cap != capacity) {
            capacity = cap;
            gaussian_ids = mtensor_empty(dev, {cap}, DType::Int32);
            packed_rgb = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_xy = mtensor_empty(dev, {cap, 2}, DType::Float32);
            packed_transmat = mtensor_empty(dev, {cap, 9}, DType::Float32);
            packed_normal_opac = mtensor_empty(dev, {cap, 4}, DType::Float32);
        }
        if (ih != img_height || iw != img_width) {
            img_height = ih; img_width = iw;
            out_img = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
            // M2.1: final_Ts widens to (T_final, M1, M2); final_idx widens to (last, median).
            final_Ts = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
            final_idx = mtensor_empty(dev, {ih, iw, 2}, DType::Int32);
            out_depth = mtensor_empty(dev, {ih, iw}, DType::Float32);
            out_normal = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
            out_alpha = mtensor_empty(dev, {ih, iw}, DType::Float32);
            out_median_depth = mtensor_empty(dev, {ih, iw}, DType::Float32);
            out_distortion = mtensor_empty(dev, {ih, iw}, DType::Float32);
        }
        if (nt != num_tiles) {
            num_tiles = nt;
            tile_bins = mtensor_empty(dev, {nt, 2}, DType::Int32);
            tile_offsets = mtensor_empty(dev, {nt}, DType::Int32);
            tile_scatter_counters = mtensor_empty(dev, {nt}, DType::Int32);
            prealloc_bins = mtensor_empty(dev, {(int64_t)nt * 2048}, DType::Int64);
        }
        if (!overflow_flag.defined()) {
            overflow_flag = mtensor_empty(dev, {1}, DType::Int32);
        }
    }

    // Lazy allocation for the SH-rest gradient — sized by frBases (depends on
    // current SH degree, which grows during training).
    void ensure_features_rest_grad(int frBases, id<MTLDevice> dev) {
        if (frBases != frBases_for_grad || !dL_dfeatures_rest.defined()) {
            frBases_for_grad = frBases;
            dL_dfeatures_rest = mtensor_empty(dev,
                {(int64_t)fwd_num_points, (int64_t)frBases, 3}, DType::Float32);
        }
    }
};
static FusedTensorCache g_tcache;

void cleanup_msplat_metal() {
    g_tcache = FusedTensorCache{};
}

// Internal forward pipeline — render-only. 2DGS dispatch.
static void forward_pipeline(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    // --- Overflow check: detect per-tile overflow (> 2048 gaussians in a tile) ---
    // Only warn once to avoid noisy output (per-tile overflow is common at >1M gaussians)
    static bool overflow_warned = false;
    static int iter_count_oc = 0;
    iter_count_oc++;
    bool num_points_changed = (num_points != g_tcache.fwd_num_points && g_tcache.fwd_num_points > 0);
    if (!overflow_warned && g_tcache.overflow_flag.defined() && g_tcache.fwd_num_points > 0
        && (num_points_changed || (iter_count_oc % 100) == 1)) {
        ctx->syncCB();
        int32_t flag_val = *g_tcache.overflow_flag.data<int32_t>();
        if (flag_val > 0) {
            fprintf(stderr, "WARNING: per-tile overflow (>2048 gaussians in a tile). "
                    "Some gaussians were dropped from overfull tiles.\n");
            overflow_warned = true;
        }
    }
    int64_t capacity = (int64_t)num_points * g_tcache.capacity_multiplier;
    uint32_t channels = 3;

    // --- Cached buffer pool: only reallocate on dimension change (densification) ---
    g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles, ctx->device);
    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;

    // --- Constants (heap-allocated for Obj-C block) ---
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 3>>(std::array<float, 3>{cam_pos[0], cam_pos[1], cam_pos[2]});
    uint32_t num_points_u32 = (uint32_t)num_points;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});

    // Periodic diagnostic: print key dimensions for roofline analysis
    static int diag_count = 0;
    diag_count++;
    if (std::getenv("BENCHMARK") && (diag_count == 100 || diag_count == 500 || diag_count == 1500)) {
            fprintf(stderr, "\n=== Roofline Dimensions (iter %d) ===\n", diag_count);
            fprintf(stderr, "  num_points:     %d\n", num_points);
            fprintf(stderr, "  capacity:       %lld (= num_points * %lld)\n", (long long)capacity, (long long)g_tcache.capacity_multiplier);
            fprintf(stderr, "  img:            %u x %u = %u pixels\n", img_width, img_height, img_width * img_height);
            fprintf(stderr, "  tiles:          %d x %d = %d\n", tile_bounds_x, tile_bounds_y, num_tiles);
            fprintf(stderr, "  SH degree:      %u (bases: %u)\n", degree, (degree + 1) * (degree + 1));
            fprintf(stderr, "  features_rest:  [%lld x %lld x %lld]\n",
                (long long)features_rest.size(0), (long long)features_rest.size(1), (long long)features_rest.size(2));
            fprintf(stderr, "  sort:           tile-local (bitonic, max 2048/tile)\n");
            fprintf(stderr, "  sort buffer:    %.1f MB (sort_pairs)\n", (double)capacity * 8.0 / 1e6);
            fprintf(stderr, "  opacities:      [%lld]\n", (long long)opacities.size(0));
            fprintf(stderr, "===========================\n\n");
    }

    // ===== 2DGS forward dispatch =====
    // The codebase committed to 2DGS in Phase 2b.3.2 (5/N) — kScaleDim=2,
    // float2 scales storage. The 3DGS dispatch lambdas and env-var gate
    // were removed in the post-Milestone-1 cleanup. forward_pipeline is
    // called only from msplat_render (compute_loss=false); training has
    // no forward path yet (deferred to Milestone 2).

    auto encode_proj_sh_2dgs = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_2dgs_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_2dgs_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13);
        ENC_BUF(enc, g_tcache.transMats, 14);          // replaces 3DGS `conics`
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_BUF(enc, g_tcache.normal_opacity, 16);     // float4: (n.xyz, raw opac logit)
        ENC_BUF(enc, opacities, 17);                   // raw logit-space, passed through to normal_opacity.w
        ENC_SCALAR(enc, degree, 18); ENC_SCALAR(enc, degrees_to_use, 19);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:20];
        ENC_BUF(enc, features_dc, 21); ENC_BUF(enc, features_rest, 22);
        ENC_BUF(enc, colors, 23); ENC_BUF(enc, aabb, 24);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_prefix_map_2dgs = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_tiles_u32 = (uint32_t)num_tiles;
        // 1. scatter_to_prealloc_bins — identical to 3DGS path.
        {
            NSUInteger tpg = MIN(ctx->scatter_to_prealloc_bins_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->scatter_to_prealloc_bins_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1); ENC_BUF(enc, depths, 2);
            ENC_BUF(enc, radii_out, 3); ENC_BUF(enc, aabb, 4);
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:5];
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 6);
            ENC_BUF(enc, g_tcache.prealloc_bins, 7);
            ENC_BUF(enc, g_tcache.overflow_flag, 8);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 2. prefix_sum — identical to 3DGS path.
        {
            NSUInteger tg2 = MIN(ctx->prefix_sum_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
            [enc setComputePipelineState:ctx->prefix_sum_kernel_cpso];
            ENC_SCALAR(enc, num_tiles_u32, 0); ENC_BUF(enc, g_tcache.tile_scatter_counters, 1); ENC_BUF(enc, g_tcache.tile_offsets, 2);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 3. bitonic_sort_per_tile_2dgs — swaps in transMats + normal_opacity inputs;
        //    emits the 2DGS sorted-surfel pack (xy / transmat / normal+sigmoid(opac) / rgb).
        {
            [enc setComputePipelineState:ctx->bitonic_sort_per_tile_2dgs_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0); ENC_BUF(enc, g_tcache.tile_scatter_counters, 1);
            ENC_BUF(enc, g_tcache.prealloc_bins, 2);
            ENC_BUF(enc, gaussian_ids, 3);
            ENC_SCALAR(enc, num_tiles_u32, 4);
            ENC_BUF(enc, xys, 5);
            ENC_BUF(enc, g_tcache.transMats, 6);           // replaces 3DGS `conics`
            ENC_BUF(enc, g_tcache.normal_opacity, 7);      // float4 with raw opac in .w
            ENC_BUF(enc, colors, 8);
            ENC_BUF(enc, g_tcache.packed_xy, 9);
            ENC_BUF(enc, g_tcache.packed_transmat, 10);
            ENC_BUF(enc, g_tcache.packed_normal_opac, 11); // .w := sigmoid(raw)
            ENC_BUF(enc, packed_rgb, 12);
            ENC_BUF(enc, tile_bins, 13);
            [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    auto encode_rast_fwd_2dgs = [&](id<MTLComputeCommandEncoder> enc) {
        // Monolithic only. Slots 0–12 are the M1 forward args; slots 13–15 are
        // the M2.1 aux outputs (alpha, median_depth, distortion) needed by the
        // backward replay + regularizer losses. background → 16, blockDim → 17.
        MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->nd_rasterize_forward_2dgs_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, g_tcache.packed_xy, 4);
        ENC_BUF(enc, g_tcache.packed_normal_opac, 5);
        ENC_BUF(enc, g_tcache.packed_transmat, 6);
        ENC_BUF(enc, packed_rgb, 7);
        ENC_BUF(enc, final_Ts, 8); ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, out_img, 10);
        ENC_BUF(enc, g_tcache.out_depth, 11);
        ENC_BUF(enc, g_tcache.out_normal, 12);
        ENC_BUF(enc, g_tcache.out_alpha, 13);
        ENC_BUF(enc, g_tcache.out_median_depth, 14);
        ENC_BUF(enc, g_tcache.out_distortion, 15);
        ENC_BUF(enc, background, 16);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:17];
        [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:tg_size];
    };

    {
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        dispatch_sync(ctx->d_queue, ^(){
            // Blit-zero buffers that accumulate across gaussians (must be GPU-side
            // to avoid racing with previous CB's reads on pipelined execution).
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            [blit fillBuffer:g_tcache.overflow_flag.buffer() range:NSMakeRange(0, g_tcache.overflow_flag.nbytes()) value:0];
            [blit fillBuffer:g_tcache.tile_scatter_counters.buffer() range:NSMakeRange(0, g_tcache.tile_scatter_counters.nbytes()) value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            assert(encoder && "Failed to create compute command encoder");

            encode_proj_sh_2dgs(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_prefix_map_2dgs(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd_2dgs(encoder);

            [encoder endEncoding];
        });
    }

    // All outputs are in g_tcache — no return needed
}

MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
) {
    forward_pipeline(num_points, means3d, scales, glob_scale,
        quats, viewmat, projmat, fx, fy, cx, cy,
        img_height, img_width, tile_bounds, clip_thresh,
        degree, degrees_to_use, cam_pos, features_dc, features_rest,
        opacities, background);
    return g_tcache.out_img;
}

// 2DGS side outputs from the most recent msplat_render. All written by
// nd_rasterize_forward_2dgs_kernel into g_tcache.
MTensor msplat_last_out_depth()        { return g_tcache.out_depth; }
MTensor msplat_last_out_normal()       { return g_tcache.out_normal; }
MTensor msplat_last_out_alpha()        { return g_tcache.out_alpha; }
MTensor msplat_last_out_median_depth() { return g_tcache.out_median_depth; }
MTensor msplat_last_out_distortion()   { return g_tcache.out_distortion; }


// ============================================================================
// GPU-native densification (v34 Phase 3)
// Entire classify → grow → cull → compact pipeline in one compute encoder.
// Returns new num_active after densification.
// ============================================================================
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
) {
    MetalContext* ctx = get_global_context();

    // Worst case: each of N gaussians splits (2 children) + dups (1 copy) = 3N
    int worst_case = 3 * N;
    assert(worst_case <= buf_capacity && "gpu_densify: 3*N exceeds buf_capacity");

    float log_size_fac = std::log(1.6f);

    // Strides for each of the 18 buffers (6 params + 12 optimizer states)
    // Order: means(3), scales(3), quats(4), featuresDc(3), featuresRest(fr_stride), opacities(1)
    int strides[6] = {3, 3, 4, 3, fr_stride, 1};
    int max_stride = fr_stride;  // featuresRest has the largest stride

    // Collect all 18 buffers in order for compact loops (std::array for block capture)
    std::array<MTensor*, 18> all_bufs = {{
        &means_buf, &scales_buf, &quats_buf, &featuresDc_buf, &featuresRest_buf, &opacities_buf,
        &adam_exp_avg_buf[0], &adam_exp_avg_buf[1], &adam_exp_avg_buf[2],
        &adam_exp_avg_buf[3], &adam_exp_avg_buf[4], &adam_exp_avg_buf[5],
        &adam_exp_avg_sq_buf[0], &adam_exp_avg_sq_buf[1], &adam_exp_avg_sq_buf[2],
        &adam_exp_avg_sq_buf[3], &adam_exp_avg_sq_buf[4], &adam_exp_avg_sq_buf[5]
    }};
    std::array<int, 18> all_strides = {{
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1
    }};

    uint32_t N_u32 = (uint32_t)N;
    uint32_t K = (uint32_t)((N + 1023) / 1024);  // threadgroups for prefix sum on N elements
    int check_screen_int = check_screen;
    int check_huge_int = check_huge;

    id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
    assert(command_buffer && "Failed to retrieve command buffer reference");

    dispatch_sync(ctx->d_queue, ^(){
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        assert(enc && "Failed to create compute command encoder");

        // ---- Stage 1: Classify (split/dup) ----
        {
            NSUInteger tpg = MIN(ctx->densify_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, xys_grad_norm, 1);
            ENC_BUF(enc, vis_counts, 2);
            ENC_BUF(enc, scales_buf, 3);
            ENC_BUF(enc, max_2d_size, 4);
            ENC_SCALAR(enc, half_max_dim, 5);
            ENC_SCALAR(enc, grad_thresh, 6);
            ENC_SCALAR(enc, size_thresh, 7);
            ENC_SCALAR(enc, screen_thresh, 8);
            ENC_SCALAR(enc, check_screen_int, 9);
            ENC_BUF(enc, split_flag, 10);
            ENC_BUF(enc, dup_flag, 11);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 2: Prefix sum on split_flag → split_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 3: Prefix sum on dup_flag → dup_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 4: Append split children ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_split_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_split_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2);
            ENC_BUF(enc, random_samples, 3);
            ENC_SCALAR(enc, log_size_fac, 4);
            ENC_BUF(enc, means_buf, 5);
            ENC_BUF(enc, scales_buf, 6);
            ENC_BUF(enc, quats_buf, 7);
            ENC_BUF(enc, featuresDc_buf, 8);
            ENC_BUF(enc, featuresRest_buf, 9);
            ENC_BUF(enc, opacities_buf, 10);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 11);
            ENC_BUF(enc, adam_exp_avg_buf[0], 12);
            ENC_BUF(enc, adam_exp_avg_buf[1], 13);
            ENC_BUF(enc, adam_exp_avg_buf[2], 14);
            ENC_BUF(enc, adam_exp_avg_buf[3], 15);
            ENC_BUF(enc, adam_exp_avg_buf[4], 16);
            ENC_BUF(enc, adam_exp_avg_buf[5], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 22);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 23);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 5: Append duplicates ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_dup_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_dup_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_prefix, 3);
            ENC_BUF(enc, means_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, quats_buf, 6);
            ENC_BUF(enc, featuresDc_buf, 7);
            ENC_BUF(enc, featuresRest_buf, 8);
            ENC_BUF(enc, opacities_buf, 9);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 10);
            ENC_BUF(enc, adam_exp_avg_buf[0], 11);
            ENC_BUF(enc, adam_exp_avg_buf[1], 12);
            ENC_BUF(enc, adam_exp_avg_buf[2], 13);
            ENC_BUF(enc, adam_exp_avg_buf[3], 14);
            ENC_BUF(enc, adam_exp_avg_buf[4], 15);
            ENC_BUF(enc, adam_exp_avg_buf[5], 16);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 22);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 6: Cull classify (on post-growth population) ----
        // Dispatch worst_case threads; kernel reads N_new from prefix sums
        {
            uint32_t wc = (uint32_t)worst_case;
            NSUInteger tpg = MIN(ctx->densify_cull_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)worst_case);
            [enc setComputePipelineState:ctx->densify_cull_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_prefix, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_flag, 3);
            ENC_BUF(enc, opacities_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, max_2d_size, 6);
            ENC_SCALAR(enc, cull_alpha_thresh, 7);
            ENC_SCALAR(enc, cull_scale_thresh, 8);
            ENC_SCALAR(enc, cull_screen_size, 9);
            ENC_SCALAR(enc, check_huge_int, 10);
            ENC_SCALAR(enc, check_screen_int, 11);
            ENC_BUF(enc, keep_flag, 12);
            [enc dispatchThreads:MTLSizeMake(worst_case, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 7: Prefix sum on keep_flag → keep_prefix ----
        // Over worst_case elements (includes padding zeros for unused slots)
        {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t K2 = (uint32_t)((worst_case + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t K2 = (uint32_t)((worst_case + 1023) / 1024);
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, keep_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 8: Compact scatter (18 buffers → scratch) ----
        // For each buffer: scatter kept elements into compact_scratch
        // Then copy back. We reuse compact_scratch at different offsets per stride.
        for (int b = 0; b < 18; b++) {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t stride_u32 = (uint32_t)all_strides[b];
            uint32_t total_threads = wc * stride_u32;
            NSUInteger tpg = MIN(ctx->compact_scatter_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)total_threads);
            [enc setComputePipelineState:ctx->compact_scatter_kernel_cpso];
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:0];
            ENC_BUF(enc, compact_scratch, 1);
            ENC_BUF(enc, keep_prefix, 2);
            ENC_BUF(enc, keep_flag, 3);
            ENC_SCALAR(enc, wc, 4);
            ENC_SCALAR(enc, stride_u32, 5);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Copy back from scratch to buffer
            uint32_t last_idx = wc - 1;
            [enc setComputePipelineState:ctx->compact_copy_back_kernel_cpso];
            ENC_BUF(enc, compact_scratch, 0);
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:1];
            ENC_BUF(enc, keep_prefix, 2);
            ENC_SCALAR(enc, last_idx, 3);
            ENC_SCALAR(enc, stride_u32, 4);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc endEncoding];
    });

    // Single GPU→CPU sync: read new_count from keep_prefix[worst_case - 1]
    ctx->syncCB();
    int new_count = keep_prefix.data<int32_t>()[worst_case - 1];
    return new_count;
}
