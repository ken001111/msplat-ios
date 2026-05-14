#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8
#define RAST_BLOCK_SIZE (RAST_BLOCK_X * RAST_BLOCK_Y)
#define CHANNELS 3
#define MAX_REGISTER_CHANNELS 3

// Scale-vector dim. 3 for 3DGS (axis-aligned ellipsoid), 2 for 2DGS (in-plane
// disc). Kept in lockstep with core/include/model.hpp's constexpr int kScaleDim;
// Phase 2b.3.2 (5/N) flipped both sides to 2 — codebase now commits to 2DGS.
constant int kScaleDim = 2;

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
constant float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

inline uint num_sh_bases(const uint degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

inline float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5;
}

inline void get_bbox(
    const float2 center,
    const float2 dims,
    const int3 img_size,
    thread uint2 &bb_min,
    thread uint2 &bb_max
) {
    // Clamp axis-aligned bounding box to valid range [0, img_size).
    // Returns inclusive min, exclusive max.
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline void get_tile_bbox(
    const float2 pix_center,
    const float2 pix_radius,
    const int3 tile_bounds,
    thread uint2 &tile_min,
    thread uint2 &tile_max
) {
    // Convert pixel-space center/radius to tile coordinates and compute AABB.
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius.x / (float)BLOCK_X, pix_radius.y / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// Affine transform: mat (row-major 4x3) applied to point p.
inline float3 transform_4x3(constant float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// Full 4x4 row-major transform, returns homogeneous coordinates.
inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

// Normalized quaternion → 3x3 rotation matrix (column-major for Metal).
inline float3x3 quat_to_rotmat(const float4 quat) {
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    return float3x3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// Returns true if point is behind the near plane (should be culled).
inline bool clip_near_plane(
    const float3 p, 
    constant float *viewmat, 
    thread float3 &p_view, 
    float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline float3x3 scale_to_mat(const float3 scale, const float glob_scale) {
    float3x3 S = float3x3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// Build 3D covariance matrix from scale + quaternion: cov = R*S*S^T*R^T.
// Stores upper triangle (6 floats) since the matrix is symmetric.
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, device float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);

    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);

    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// Thread-local overload: writes cov3d to registers instead of device memory
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, thread float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// ===== 2DGS surfel helpers (Phase 2b.3.1) =====
// Port of hbb1/2d-gaussian-splatting CUDA helpers in
//   submodules/diff-surfel-rasterization/cuda_rasterizer/{auxiliary.h,forward.cu}.
// In 2DGS each gaussian is a flat disc on a tangent plane (Huang et al. 2024).
// Per-gaussian state: scale is float2 (in-plane radii), quat defines the tangent
// frame (columns: tangent_u, tangent_v, normal). The forward projection writes
// a 3x3 homography T mapping homogeneous pixel coords → homogeneous tangent uv
// coords, so the rasterizer can recover (u, v, depth) in a few FMAs per pixel.

// Low-pass filter cutoff (matches forward.cu FilterSize = sqrt(2)/2).
constant float FILTER_SIZE_2DGS = 0.707106f;
// 1 / FilterSize^2 — pre-inverted (matches forward.cu FilterInvSquare = 2.0).
constant float FILTER_INV_SQ_2DGS = 2.0f;

// Near/far clip for the distortion regularizer's normalized depth mapping
// m = far / (far - near) * (1 - near / depth). Matches auxiliary.h:near_n/far_n.
constant float NEAR_N_2DGS = 0.2f;
constant float FAR_N_2DGS = 100.0f;

// Compute the 3x3 homography T that maps (u, v, 1) in the surfel tangent
// frame → (px*w, py*w, w) in homogeneous pixel coords. Also returns the
// surfel normal in world space.
//
// Storage convention: T is returned as 3 row-vectors (one per math row).
// row 0 is the "px*w" linear functional, row 1 is "py*w", row 2 is "w".
// The forward kernel writes these three rows into the conics buffer
// sequentially as 9 floats per gaussian. The rasterizer reads them back
// to construct two homogeneous planes (k = px*row2 - row0, l = py*row2 -
// row1) whose intersection gives (u, v) via Eq. 8-10 of the paper.
//
// msplat stores projmat as a flat row-major float[16].
inline void compute_transmat_2dgs(
    const float3 p_orig,
    const float2 scale_2d,          // linear-space radii (caller has already exp'd if log-space)
    const float glob_scale,
    const float4 quat,
    constant float* projmat,        // 4x4 row-major: world → clip
    const float W,
    const float H,
    const float cx,                 // principal point x (pixels)
    const float cy,                 // principal point y (pixels)
    thread float3& out_T_row0,
    thread float3& out_T_row1,
    thread float3& out_T_row2,
    thread float3& out_normal_world
) {
    // Tangent frame in world space, scaled by per-axis surfel radii.
    float3x3 R = quat_to_rotmat(quat);
    float3 tu = R[0] * (scale_2d.x * glob_scale);    // scaled tangent_u (3-vec, world dir)
    float3 tv = R[1] * (scale_2d.y * glob_scale);    // scaled tangent_v
    out_normal_world = R[2];                          // unit normal

    // Project tu, tv (directions, w=0) and p_orig (point, w=1) to clip space.
    // projmat is row-major: row i has indices [4*i .. 4*i+3].
    float4 tu_clip = float4(
        projmat[0]*tu.x + projmat[1]*tu.y + projmat[2]*tu.z,
        projmat[4]*tu.x + projmat[5]*tu.y + projmat[6]*tu.z,
        projmat[8]*tu.x + projmat[9]*tu.y + projmat[10]*tu.z,
        projmat[12]*tu.x + projmat[13]*tu.y + projmat[14]*tu.z
    );
    float4 tv_clip = float4(
        projmat[0]*tv.x + projmat[1]*tv.y + projmat[2]*tv.z,
        projmat[4]*tv.x + projmat[5]*tv.y + projmat[6]*tv.z,
        projmat[8]*tv.x + projmat[9]*tv.y + projmat[10]*tv.z,
        projmat[12]*tv.x + projmat[13]*tv.y + projmat[14]*tv.z
    );
    float4 p_clip = float4(
        projmat[0]*p_orig.x + projmat[1]*p_orig.y + projmat[2]*p_orig.z + projmat[3],
        projmat[4]*p_orig.x + projmat[5]*p_orig.y + projmat[6]*p_orig.z + projmat[7],
        projmat[8]*p_orig.x + projmat[9]*p_orig.y + projmat[10]*p_orig.z + projmat[11],
        projmat[12]*p_orig.x + projmat[13]*p_orig.y + projmat[14]*p_orig.z + projmat[15]
    );

    // NDC → homogeneous pixel: pixhomog = (0.5*W*c.x + (cx-0.5)*c.w,
    //                                       0.5*H*c.y + (cy-0.5)*c.w,
    //                                       c.w).
    // (The -0.5 offset matches msplat's existing ndc2pix() helper.)
    float halfW = 0.5f * W;
    float halfH = 0.5f * H;
    float bx = cx - 0.5f;
    float by = cy - 0.5f;

    float3 col_u = float3(halfW*tu_clip.x + bx*tu_clip.w,
                          halfH*tu_clip.y + by*tu_clip.w,
                          tu_clip.w);
    float3 col_v = float3(halfW*tv_clip.x + bx*tv_clip.w,
                          halfH*tv_clip.y + by*tv_clip.w,
                          tv_clip.w);
    float3 col_o = float3(halfW*p_clip.x  + bx*p_clip.w,
                          halfH*p_clip.y  + by*p_clip.w,
                          p_clip.w);

    // Math T satisfies T · (u, v, 1) = (px*w, py*w, w) componentwise.
    //   row 0: linear functional yielding px*w
    //   row 1: yielding py*w
    //   row 2: yielding w
    // row_i = (col_u.<i>, col_v.<i>, col_o.<i>) for i in {x, y, z}.
    out_T_row0 = float3(col_u.x, col_v.x, col_o.x);
    out_T_row1 = float3(col_u.y, col_v.y, col_o.y);
    out_T_row2 = float3(col_u.z, col_v.z, col_o.z);
}

// Screen-space AABB of a 2DGS surfel given its T matrix (as 3 rows).
// Mirrors compute_aabb in hbb1's forward.cu — solves the ellipse equation
// s^T (T_row0_T_row0 + T_row1_T_row1 - cutoff^2 T_row2_T_row2) s = 0 in
// closed form to get the screen-space center and per-axis half-extent.
// Returns false if the surfel is degenerate (zero w-row).
inline bool compute_aabb_2dgs(
    const float3 T_row0,
    const float3 T_row1,
    const float3 T_row2,
    const float cutoff,
    thread float2& out_point_image,
    thread float2& out_extent
) {
    // t = (k², k², -1)  where k = cutoff
    float3 t = float3(cutoff * cutoff, cutoff * cutoff, -1.0f);
    float d = dot(t, T_row2 * T_row2);
    if (d == 0.0f) return false;
    float3 f = (1.0f / d) * t;

    float2 p = float2(
        dot(f, T_row0 * T_row2),
        dot(f, T_row1 * T_row2)
    );

    float2 h0 = p * p - float2(
        dot(f, T_row0 * T_row0),
        dot(f, T_row1 * T_row1)
    );

    float2 h = sqrt(max(float2(1e-4f, 1e-4f), h0));
    out_point_image = p;
    out_extent = h;
    return true;
}

// Project 3D covariance to 2D via EWA splatting.
// Takes pre-computed view-space position; exploits J sparsity (5/9 nonzero).
float3 project_cov3d_ewa(
    device float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 p_view
) {
    // Clamp view-space position to avoid extreme covariance at FOV edges
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    // T = J * W where J has only 5 nonzero entries.
    // Instead of full 3x3 matmul, compute T rows directly.
    // T_row0 = j00 * M_row0 + j20 * M_row2 (viewmat is row-major)
    // T_row1 = j11 * M_row1 + j21 * M_row2
    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2;
    float j21 = -fy * p_view.y * rz2;

    float3 mr0 = float3(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1 = float3(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2 = float3(viewmat[8], viewmat[9], viewmat[10]);

    float3 t0 = j00 * mr0 + j20 * mr2;  // T row 0
    float3 t1 = j11 * mr1 + j21 * mr2;  // T row 1

    // cov2d = T * V * T^T, upper-left 2x2 only (3 values)
    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    return float3(dot(tv0, t0) + 0.3f, dot(tv0, t1), dot(tv1, t1) + 0.3f);
}

// Thread-local overload: reads cov3d from registers
float3 project_cov3d_ewa(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2;
    float j21 = -fy * p_view.y * rz2;

    float3 mr0 = float3(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1 = float3(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2 = float3(viewmat[8], viewmat[9], viewmat[10]);

    float3 t0 = j00 * mr0 + j20 * mr2;
    float3 t1 = j11 * mr1 + j21 * mr2;

    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    return float3(dot(tv0, t0) + 0.3f, dot(tv0, t1), dot(tv1, t1) + 0.3f);
}

inline bool compute_cov2d_bounds(
    const float3 cov2d, 
    thread float3 &conic, 
    thread float &radius
) {
    // Invert 2x2 covariance (upper triangle in cov2d.xyz) to get the conic,
    // and compute the gaussian's screen-space radius from eigenvalues (3-sigma).
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // Conic = inverse of 2x2 covariance
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float disc = sqrt(max(0.1f, b * b - det));
    // 3-sigma radius from the larger eigenvalue
    radius = ceil(3.f * sqrt(b + disc));
    return true;
}

// Project 3D point to pixel coordinates via projection matrix.
inline float2 project_pix(
    constant float *mat, const float3 p, const uint2 img_size, const float2 pp
) {
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, (int)img_size.x, pp.x), ndc2pix(p_proj.y, (int)img_size.y, pp.y)
    };
}

// Metal pads vector types in arrays (e.g. float3 → 16 bytes). These helpers
// read/write contiguous packed data by indexing into the underlying scalar buffer.

inline int2 read_packed_int2(constant int* arr, int idx) {
    return int2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_int2(device int* arr, int idx, int2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline void write_packed_int2x(device int* arr, int idx, int x) {
    arr[2*idx] = x;
}

inline void write_packed_int2y(device int* arr, int idx, int y) {
    arr[2*idx+1] = y;
}

inline float2 read_packed_float2(constant float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline float2 read_packed_float2(device float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_float2(device float* arr, int idx, float2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline int3 read_packed_int3(constant int* arr, int idx) {
    return int3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_int3(device int* arr, int idx, int3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float3 read_packed_float3(constant float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline float3 read_packed_float3(device float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline float3 read_packed_float3(device const float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_float3(device float* arr, int idx, float3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float4 read_packed_float4(constant float* arr, int idx) {
    return float4(arr[4*idx], arr[4*idx+1], arr[4*idx+2], arr[4*idx+3]);
}

inline void write_packed_float4(device float* arr, int idx, float4 val) {
    arr[4*idx] = val.x;
    arr[4*idx+1] = val.y;
    arr[4*idx+2] = val.z;
    arr[4*idx+3] = val.w;
}

// Forward projection: one thread per gaussian. Computes 2D position, conic, radius.


// ===== 2DGS forward rasterizer (Phase 2b.3.1, step 3) =====
// Port of hbb1/2d-gaussian-splatting renderCUDA (forward.cu:256-457). Replaces
// the per-pixel screen-space conic sigma with a ray-plane intersection in the
// surfel's tangent frame; outputs an extra depth map and an alpha-weighted
// world-space normal map (out_others in the reference, split here for cleaner
// buffer allocation).
//
// Compositing logic (1/255 alpha cutoff, T < 1e-4 early exit) is identical to
// the 3DGS path — only the per-gaussian (u, v, depth) recovery differs.
//
// In addition to the inference outputs (out_img, out_depth, out_normal) the
// kernel accumulates the backward-replay state that rasterize_backward_2dgs
// needs: M1 / M2 (running depth moments in normalized depth space), the
// per-pixel accumulated alpha, the median-depth contributor, and the depth-
// distortion regularizer. All eight outputs are stored as separate Float32
// tensors rather than the reference's packed [H, W, 7] aux buffer — clearer
// to reason about on the host side.
kernel void nd_rasterize_forward_2dgs_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int* tile_bins,                // int2 (start, end) per tile
    constant float* packed_xy,              // float2 per sorted-gaussian — surfel center
    constant float* packed_normal_opac,     // float4 per sorted-gaussian — (n.x, n.y, n.z, sigmoid(opac))
    constant float* packed_transmat,        // 9 floats per sorted-gaussian — 3 rows of T
    constant float* packed_rgb,             // float3 per sorted-gaussian — raw SH output (NOT clamped)
    device float* final_Ts,                 // float3 per pixel — (T_final, M1, M2) (M2.1)
    device int* final_index,                // int2 per pixel — (last_contributor, median_contributor) (M2.1)
    device float* out_img,                  // float3 per pixel (RGB)
    device float* out_depth,                // float per pixel — alpha-weighted depth (DEPTH_OFFSET)
    device float* out_normal,               // float3 per pixel — alpha-weighted normal (NORMAL_OFFSET)
    device float* out_alpha,                // float per pixel — 1 - T_final (ALPHA_OFFSET) (M2.1)
    device float* out_median_depth,         // float per pixel — depth where T crosses 0.5 (MIDDEPTH_OFFSET) (M2.1)
    device float* out_distortion,           // float per pixel — depth-distortion regularizer (DISTORTION_OFFSET) (M2.1)
    constant float* background,
    constant uint2& blockDim,
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]]
) {
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * (int)img_size.x + j;

    const bool inside = (i < (int)img_size.y && j < (int)img_size.x);

    int2 range = read_packed_int2(tile_bins, tile_id);
    const int num_batches = (range.y - range.x + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    // Threadgroup shared memory: per-surfel state for one batch.
    // 64 surfels × (xy + normal_opac + Tu + Tv + Tw + rgb) ≈ 5.5 KB,
    // well under Apple GPU family 8/9 threadgroup-memory limits.
    threadgroup float2 xy_batch[RAST_BLOCK_SIZE];
    threadgroup float4 normal_opac_batch[RAST_BLOCK_SIZE];
    threadgroup float3 Tu_batch[RAST_BLOCK_SIZE];
    threadgroup float3 Tv_batch[RAST_BLOCK_SIZE];
    threadgroup float3 Tw_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];

    float T = 1.f;
    float3 pix_color = {0.f, 0.f, 0.f};
    float3 pix_normal = {0.f, 0.f, 0.f};
    float pix_depth = 0.f;

    // Distortion regularizer state (forward.cu lines 397-411).
    // m = far / (far - near) * (1 - near / depth) — depth normalized to [0, 1].
    // M1 = Σ m * w_i,  M2 = Σ m^2 * w_i,  distortion = Σ (m^2 A + M2 - 2 m M1) w_i
    float M1 = 0.f;
    float M2 = 0.f;
    float distortion = 0.f;
    float median_depth = 0.f;
    int median_contributor = -1;

    // Counter semantics: `contributor` is a 1-based per-pixel iter counter
    // that increments at the TOP of each surfel iteration (matches reference
    // forward.cu line 303-347). `last_contributor` records the value of
    // `contributor` after the most-recent successful blend. `median_contributor`
    // is set during the T>0.5 blend. These semantics are what the backward's
    // `if (contributor >= last_contributor) continue;` check expects (reference
    // backward.cu line 199-278). Earlier this code stored sorted-list-absolute
    // indices, which silently corrupted gradient propagation.
    int contributor = 0;
    int last_contributor = 0;
    bool done = false;

    for (int b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int batch_start = range.x + RAST_BLOCK_SIZE * b;
        int idx_load = batch_start + (int)tr;
        if (idx_load < range.y) {
            xy_batch[tr]            = read_packed_float2(packed_xy, idx_load);
            normal_opac_batch[tr]   = read_packed_float4(packed_normal_opac, idx_load);
            // Read T as 3 contiguous float3 rows from the packed transmat buffer.
            uint t_base = (uint)idx_load * 9;
            Tu_batch[tr] = float3(packed_transmat[t_base + 0], packed_transmat[t_base + 1], packed_transmat[t_base + 2]);
            Tv_batch[tr] = float3(packed_transmat[t_base + 3], packed_transmat[t_base + 4], packed_transmat[t_base + 5]);
            Tw_batch[tr] = float3(packed_transmat[t_base + 6], packed_transmat[t_base + 7], packed_transmat[t_base + 8]);
            // packed_rgb stores the raw SH output (pre-clamp), matches 3DGS path
            const float3 raw_c = read_packed_float3(packed_rgb, idx_load);
            rgbs_batch[tr] = max(raw_c + 0.5f, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (done || !inside) continue;

        int batch_size = min(RAST_BLOCK_SIZE, range.y - batch_start);

        for (int t = 0; t < batch_size; ++t) {
            // 1-based per-pixel iter counter increments BEFORE any skip — matches
            // reference forward.cu line 347.
            contributor++;

            // Build two homogeneous planes from T's rows. (forward.cu Eq. 8)
            const float3 Tu = Tu_batch[t];
            const float3 Tv = Tv_batch[t];
            const float3 Tw = Tw_batch[t];
            const float3 k = px * Tw - Tu;
            const float3 l = py * Tw - Tv;
            // Their cross product is the ray; perspective-divide → (u, v).
            const float3 ray = cross(k, l);
            if (ray.z == 0.0f) continue;
            const float inv_z = 1.0f / ray.z;
            const float2 uv = float2(ray.x * inv_z, ray.y * inv_z);

            // 2DGS sigma — squared distance in the tangent frame (Eq. 10).
            const float rho3d = uv.x * uv.x + uv.y * uv.y;

            // Low-pass filter: bound by squared screen-space distance.
            const float2 xy = xy_batch[t];
            const float2 d_screen = float2(xy.x - px, xy.y - py);
            const float rho2d = FILTER_INV_SQ_2DGS * (d_screen.x * d_screen.x + d_screen.y * d_screen.y);
            const float rho = min(rho3d, rho2d);

            // Per-pixel depth at the ray-plane intersection (forward.cu line 369).
            const float depth = uv.x * Tw.x + uv.y * Tw.y + Tw.z;
            if (depth < NEAR_N_2DGS) continue;

            const float power = -0.5f * rho;
            if (power > 0.0f) continue;

            const float4 nor_o = normal_opac_batch[t];
            const float opac = nor_o.w;
            const float alpha = min(0.99f, opac * exp(power));
            if (alpha < 1.0f / 255.0f) continue;

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                // Early-exit surfel did NOT blend → don't update last_contributor.
                // (Reference forward.cu line 390-394 does `done=true; continue;`,
                // which leaves contributor incremented but last_contributor unchanged.)
                done = true;
                break;
            }

            const float w = alpha * T;

            // Distortion regularizer state — accumulate BEFORE updating M1/M2
            // so the cross-term uses the prior partial sums (matches Eq. in
            // 2DGS paper appendix and forward.cu lines 400-405).
            const float A = 1.f - T;
            const float m = FAR_N_2DGS / (FAR_N_2DGS - NEAR_N_2DGS) * (1.f - NEAR_N_2DGS / depth);
            distortion += (m * m * A + M2 - 2.f * m * M1) * w;
            M1 = fma(m, w, M1);
            M2 = fma(m * m, w, M2);

            // Median: capture the first surfel that pushes T below 0.5.
            if (T > 0.5f) {
                median_depth = depth;
                median_contributor = contributor;
            }

            // RGB accumulation
            pix_color = fma(rgbs_batch[t], w, pix_color);
            // Alpha-weighted depth (DEPTH_OFFSET output).
            pix_depth = fma(depth, w, pix_depth);
            // Alpha-weighted world-space normal (NORMAL_OFFSET output).
            pix_normal = fma(float3(nor_o.x, nor_o.y, nor_o.z), w, pix_normal);

            T = next_T;
            last_contributor = contributor;
        }
    }

    if (inside) {
        // final_Ts now packs (T, M1, M2) as float3 per pixel.
        final_Ts[3 * pix_id + 0] = T;
        final_Ts[3 * pix_id + 1] = M1;
        final_Ts[3 * pix_id + 2] = M2;
        // final_index now packs (last_contributor, median_contributor) as int2 per pixel.
        final_index[2 * pix_id + 0] = last_contributor;
        final_index[2 * pix_id + 1] = median_contributor;

        // Color: composite with background, NO saturate — reference forward.cu
        // line 434 doesn't clamp, and clamping requires masking dL_dpixel in
        // the backward for the clipped channels. Match the reference exactly.
        float3 bg = float3(background[0], background[1], background[2]);
        float3 final_rgb = fma(bg, T, pix_color);
        out_img[CHANNELS * pix_id + 0] = final_rgb.x;
        out_img[CHANNELS * pix_id + 1] = final_rgb.y;
        out_img[CHANNELS * pix_id + 2] = final_rgb.z;

        // Depth: alpha-weighted accumulation. Background contributes 0.
        out_depth[pix_id] = pix_depth;

        // Normal: alpha-weighted world-space accumulation.
        out_normal[3 * pix_id + 0] = pix_normal.x;
        out_normal[3 * pix_id + 1] = pix_normal.y;
        out_normal[3 * pix_id + 2] = pix_normal.z;

        // Aux outputs for backward replay + regularizer losses (M2.1).
        out_alpha[pix_id]        = 1.f - T;
        out_median_depth[pix_id] = median_depth;
        out_distortion[pix_id]   = distortion;
    }
}

void sh_coeffs_to_color(
    const uint degree,
    const float3 viewdir,
    constant float *dc_coeffs,
    constant float *rest_coeffs,
    device float *colors
) {
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * dc_coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    // viewdir is already normalized by caller (normalize() in project_and_sh_forward_kernel etc.)
    float x = viewdir.x;
    float y = viewdir.y;
    float z = viewdir.z;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * rest_coeffs[0 * CHANNELS + c] +
                              z * rest_coeffs[1 * CHANNELS + c] -
                              x * rest_coeffs[2 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * rest_coeffs[3 * CHANNELS + c] +
             SH_C2[1] * yz * rest_coeffs[4 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * rest_coeffs[5 * CHANNELS + c] +
             SH_C2[3] * xz * rest_coeffs[6 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * rest_coeffs[7 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * rest_coeffs[8 * CHANNELS + c] +
             SH_C3[1] * xy * z * rest_coeffs[9 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * rest_coeffs[10 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 rest_coeffs[11 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * rest_coeffs[12 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * rest_coeffs[13 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * rest_coeffs[14 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * rest_coeffs[15 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * rest_coeffs[16 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * rest_coeffs[17 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * rest_coeffs[18 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 rest_coeffs[19 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * rest_coeffs[20 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 rest_coeffs[21 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * rest_coeffs[22 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 rest_coeffs[23 * CHANNELS + c]);
    }
}

void sh_coeffs_to_color_vjp(
    const uint degree,
    const float3 viewdir,
    constant float *v_colors,
    device float *v_dc_coeffs,
    device float *v_rest_coeffs
) {
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_dc_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    // viewdir is already normalized by caller
    float x = viewdir.x;
    float y = viewdir.y;
    float z = viewdir.z;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;

    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_rest_coeffs[0 * CHANNELS + c] = v1 * v_colors[c];
        v_rest_coeffs[1 * CHANNELS + c] = v2 * v_colors[c];
        v_rest_coeffs[2 * CHANNELS + c] = v3 * v_colors[c];
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_rest_coeffs[3 * CHANNELS + c] = v4 * v_colors[c];
        v_rest_coeffs[4 * CHANNELS + c] = v5 * v_colors[c];
        v_rest_coeffs[5 * CHANNELS + c] = v6 * v_colors[c];
        v_rest_coeffs[6 * CHANNELS + c] = v7 * v_colors[c];
        v_rest_coeffs[7 * CHANNELS + c] = v8 * v_colors[c];
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_rest_coeffs[8 * CHANNELS + c] = v9 * v_colors[c];
        v_rest_coeffs[9 * CHANNELS + c] = v10 * v_colors[c];
        v_rest_coeffs[10 * CHANNELS + c] = v11 * v_colors[c];
        v_rest_coeffs[11 * CHANNELS + c] = v12 * v_colors[c];
        v_rest_coeffs[12 * CHANNELS + c] = v13 * v_colors[c];
        v_rest_coeffs[13 * CHANNELS + c] = v14 * v_colors[c];
        v_rest_coeffs[14 * CHANNELS + c] = v15 * v_colors[c];
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_rest_coeffs[15 * CHANNELS + c] = v16 * v_colors[c];
        v_rest_coeffs[16 * CHANNELS + c] = v17 * v_colors[c];
        v_rest_coeffs[17 * CHANNELS + c] = v18 * v_colors[c];
        v_rest_coeffs[18 * CHANNELS + c] = v19 * v_colors[c];
        v_rest_coeffs[19 * CHANNELS + c] = v20 * v_colors[c];
        v_rest_coeffs[20 * CHANNELS + c] = v21 * v_colors[c];
        v_rest_coeffs[21 * CHANNELS + c] = v22 * v_colors[c];
        v_rest_coeffs[22 * CHANNELS + c] = v23 * v_colors[c];
        v_rest_coeffs[23 * CHANNELS + c] = v24 * v_colors[c];
    }
}



// Compute dL/d(viewdir) from SH coefficients + dL/dRGB. Mirrors reference
// backward.cu computeColorFromSH lines 36-125 (the dRGBdx/dRGBdy/dRGBdz
// accumulation) but reads our packed-features layout: features_dc[i*3+c] for
// the DC term, features_rest[(k-1)*CHANNELS+c] for basis k = 1..total-1.
// The viewdir gradient is needed because color depends on the normalized
// direction from p_orig to cam_pos, and that direction depends on p_orig —
// without this term, mean3D never gets the "view-dependent color says I
// should move there" signal and trained gaussians converge to wrong 3D
// positions (low photometric loss but unrecognizable mesh).
inline float3 sh_color_dL_ddir(
    const uint degree,
    const float3 viewdir,
    constant float *features_rest,    // [frBases, CHANNELS] = (k - 1) * 3 + c
    constant float *v_colors          // dL/dRGB (length CHANNELS)
) {
    if (degree < 1) return float3(0);

    const float x = viewdir.x, y = viewdir.y, z = viewdir.z;

    // Per-channel dRGBdx/dRGBdy/dRGBdz, length CHANNELS.
    float dRGBdx[CHANNELS] = {0};
    float dRGBdy[CHANNELS] = {0};
    float dRGBdz[CHANNELS] = {0};

    // Degree-1 contributions: basis k=1..3, features_rest indices 0..2.
    // sh[1] = -SH_C1*y → dRGBdy contribution = -SH_C1 * sh[1] (= features_rest[0])
    // sh[2] =  SH_C1*z → dRGBdz contribution =  SH_C1 * sh[2] (= features_rest[1])
    // sh[3] = -SH_C1*x → dRGBdx contribution = -SH_C1 * sh[3] (= features_rest[2])
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        dRGBdx[c] = -SH_C1 * features_rest[2 * CHANNELS + c];
        dRGBdy[c] = -SH_C1 * features_rest[0 * CHANNELS + c];
        dRGBdz[c] =  SH_C1 * features_rest[1 * CHANNELS + c];
    }

    if (degree >= 2) {
        const float xx = x*x, yy = y*y, zz = z*z;
        #pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
            const float s4 = features_rest[3 * CHANNELS + c];
            const float s5 = features_rest[4 * CHANNELS + c];
            const float s6 = features_rest[5 * CHANNELS + c];
            const float s7 = features_rest[6 * CHANNELS + c];
            const float s8 = features_rest[7 * CHANNELS + c];
            dRGBdx[c] += SH_C2[0]*y*s4 + SH_C2[2]*(-2.f*x)*s6 + SH_C2[3]*z*s7 + SH_C2[4]*(2.f*x)*s8;
            dRGBdy[c] += SH_C2[0]*x*s4 + SH_C2[1]*z*s5 + SH_C2[2]*(-2.f*y)*s6 + SH_C2[4]*(-2.f*y)*s8;
            dRGBdz[c] += SH_C2[1]*y*s5 + SH_C2[2]*(4.f*z)*s6 + SH_C2[3]*x*s7;
        }

        if (degree >= 3) {
            #pragma unroll
            for (int c = 0; c < CHANNELS; ++c) {
                const float s9  = features_rest[8 * CHANNELS + c];
                const float s10 = features_rest[9 * CHANNELS + c];
                const float s11 = features_rest[10 * CHANNELS + c];
                const float s12 = features_rest[11 * CHANNELS + c];
                const float s13 = features_rest[12 * CHANNELS + c];
                const float s14 = features_rest[13 * CHANNELS + c];
                const float s15 = features_rest[14 * CHANNELS + c];

                dRGBdx[c] +=
                    SH_C3[0] * s9  * (6.f * x * y) +
                    SH_C3[1] * s10 * (y * z) +
                    SH_C3[2] * s11 * (-2.f * x * y) +
                    SH_C3[3] * s12 * (-6.f * x * z) +
                    SH_C3[4] * s13 * (-3.f * xx + 4.f * zz - yy) +
                    SH_C3[5] * s14 * (2.f * x * z) +
                    SH_C3[6] * s15 * (3.f * (xx - yy));

                dRGBdy[c] +=
                    SH_C3[0] * s9  * (3.f * (xx - yy)) +
                    SH_C3[1] * s10 * (x * z) +
                    SH_C3[2] * s11 * (-3.f * yy + 4.f * zz - xx) +
                    SH_C3[3] * s12 * (-6.f * y * z) +
                    SH_C3[4] * s13 * (-2.f * x * y) +
                    SH_C3[5] * s14 * (-2.f * y * z) +
                    SH_C3[6] * s15 * (-6.f * x * y);

                dRGBdz[c] +=
                    SH_C3[1] * s10 * (x * y) +
                    SH_C3[2] * s11 * (8.f * y * z) +
                    SH_C3[3] * s12 * (3.f * (2.f * zz - xx - yy)) +
                    SH_C3[4] * s13 * (8.f * x * z) +
                    SH_C3[5] * s14 * (xx - yy);
            }
        }
    }

    // Project per-channel partials onto v_colors → dL/d(viewdir).
    float3 dL_ddir = float3(0);
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        dL_ddir.x += dRGBdx[c] * v_colors[c];
        dL_ddir.y += dRGBdy[c] * v_colors[c];
        dL_ddir.z += dRGBdz[c] * v_colors[c];
    }
    return dL_ddir;
}

// Back-prop dL/d(viewdir) through viewdir = normalize(v) where v = p_orig - cam_pos.
// Returns dL/dv. Matches reference auxiliary.h:dnormvdv (float3 variant).
inline float3 dnormvdv(float3 v, float3 dv) {
    float sum2 = v.x*v.x + v.y*v.y + v.z*v.z;
    float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
    return float3(
        ((sum2 - v.x*v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32,
        (-v.x * v.y * dv.x + (sum2 - v.y*v.y) * dv.y - v.z * v.y * dv.z) * invsum32,
        (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z*v.z) * dv.z) * invsum32
    );
}

// Build (tile_id, depth) pairs for each gaussian-tile intersection.

// Find start/end offsets for each tile in the sorted intersection array.

inline int warp_reduce_all_max(int val, const int warp_size) {
    return simd_max(val);
}

inline int warp_reduce_all_or(int val, const int warp_size) {
    return simd_or(val);
}

inline float3 warpSum3(float3 val, const int warp_size, const uint lane) {
    val.x = simd_sum(val.x);
    val.y = simd_sum(val.y);
    val.z = simd_sum(val.z);
    return val;
}

inline float2 warpSum2(float2 val, const int warp_size, const uint lane) {
    val.x = simd_sum(val.x);
    val.y = simd_sum(val.y);
    return val;
}

inline float warpSum(float val, const int warp_size, const uint lane) {
    return simd_sum(val);
}



// given v_xy_pix, get v_xyz
inline float3 project_pix_vjp(
    constant float *mat, const float3 p, const uint2 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y, 0.0f};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

// compute vjp from df/d_conic to df/c_cov2d
inline void cov2d_to_conic_vjp(
    float3 conic, 
    float3 v_conic, 
    device float* v_cov2d // float3
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// Thread-local overload
inline void cov2d_to_conic_vjp(
    float3 conic,
    float3 v_conic,
    thread float* v_cov2d
) {
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// output space: 2D covariance, input space: cov3d
void project_cov3d_ewa_vjp(
    constant float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 v_cov2d,
    device float* v_mean3d,
    device float* v_cov3d,
    float3 p_view
) {
    // Apply same fov clipping as forward
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );

    float3x3 J = float3x3(
        fx * rz,                0.f,                0.f,
        0.f,                    fy * rz,            0.f,
        -fx * p_view.x * rz2,  -fy * p_view.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2;
    float fy_rz2 = fy * rz2;
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx_rz2 * v_J[2][0],
        -fy_rz2 * v_J[2][1],
        -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
            fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]
    );
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

// Thread-local overload: reads cov3d from registers, writes v_cov3d to registers
void project_cov3d_ewa_vjp(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 v_cov2d,
    device float* v_mean3d,
    thread float* v_cov3d,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );

    float3x3 J = float3x3(
        fx * rz,                0.f,                0.f,
        0.f,                    fy * rz,            0.f,
        -fx * p_view.x * rz2,  -fy * p_view.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2;
    float fy_rz2 = fy * rz2;
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx_rz2 * v_J[2][0],
        -fy_rz2 * v_J[2][1],
        -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
            fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]
    );
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

inline float4 quat_to_rotmat_vjp(const float4 quat, const float3x3 v_R) {
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const device float* v_cov3d,
    device float* v_scale, // float3
    device float* v_quat // float4
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}

// Thread-local overload: reads v_cov3d from registers
void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const thread float* v_cov3d,
    device float* v_scale,
    device float* v_quat
) {
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}



// Fused Adam optimizer kernel: single-pass update for params, exp_avg, exp_avg_sq.
// Precomputed on CPU: step_size = lr / (1 - beta1^t), bc2_sqrt = sqrt(1 - beta2^t)
// Fused per-step gradient accumulation for densification.
// Replaces ~8 MPS dispatches (boolean mask, vector_norm, index_put_, max) with 1 kernel.

kernel void fused_adam_kernel(
    device float * params [[buffer(0)]],
    device const float * grads [[buffer(1)]],
    device float * exp_avg [[buffer(2)]],
    device float * exp_avg_sq [[buffer(3)]],
    constant float & step_size [[buffer(4)]],
    constant float & beta1 [[buffer(5)]],
    constant float & beta2 [[buffer(6)]],
    constant float & bc2_sqrt [[buffer(7)]],
    constant float & eps [[buffer(8)]],
    constant uint & n [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float g = grads[tid];
    float m = fma(beta1, exp_avg[tid], (1.0f - beta1) * g);
    float v = fma(beta2, exp_avg_sq[tid], (1.0f - beta2) * g * g);

    params[tid] -= step_size * m / (sqrt(v) / bc2_sqrt + eps);

    exp_avg[tid] = m;
    exp_avg_sq[tid] = v;
}

// ===== 2DGS forward projection =====
// Port of hbb1/2d-gaussian-splatting preprocessCUDA (forward.cu:148–251).
// Reads float2 scales (in-plane radii), constructs the 3x3 homography
// T_pix2uv, writes its 9 floats to the transMats buffer, and emits a per-
// surfel float4 normal_opacity (world-space normal + raw opacity).
kernel void project_and_sh_forward_2dgs_kernel(
    // Projection args
    constant int& num_points,
    constant float* means3d,         // float3 per gaussian
    constant float* scales,          // float2 per gaussian (in-plane radii, log-space)
    constant float& glob_scale,
    constant float* quats,           // float4 per gaussian (tangent-frame rotation)
    constant float* viewmat,         // 4x4 row-major
    constant float* projmat,         // 4x4 row-major (world → clip)
    constant float4& intrins,        // (fx, fy, cx, cy)
    constant uint2& img_size,
    constant uint3& tile_bounds,
    constant float& clip_thresh,
    device float* xys,               // float2 per gaussian — screen-space surfel center
    device float* depths,            // view-space z of surfel center (sorting key)
    device int* radii,
    device float* transMats,         // 9 floats per gaussian (3 rows of T)
    device int32_t* num_tiles_hit,
    device float* normal_opacity,    // float4 per gaussian: (n.x, n.y, n.z, opacity_raw)
    constant float* opacities_raw,   // float per gaussian (logit-space; passed through to normal_opacity.w)
    // SH args (same as 3DGS path)
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float3& cam_pos,
    constant float* features_dc,
    constant float* features_rest,
    device float* colors,
    device float* aabb,              // float2 per gaussian — per-axis pixel extent
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= (uint)num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = read_packed_float3(means3d, idx);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        return;
    }

    // 2DGS state: scales is float2 in log-space; quat defines the tangent frame
    float2 s_log = read_packed_float2(scales, idx);
    float2 scale_2d = exp(s_log);
    float4 quat = read_packed_float4(quats, idx);

    // Compute homography T (3 rows) and the surfel normal.
    float3 T_row0, T_row1, T_row2;
    float3 normal_world;
    compute_transmat_2dgs(
        p_world, scale_2d, glob_scale, quat, projmat,
        (float)img_size.x, (float)img_size.y, intrins.z, intrins.w,
        T_row0, T_row1, T_row2, normal_world
    );

    // Screen-space center + extent from compute_aabb (3-sigma cutoff).
    const float cutoff = 3.0f;
    float2 center;
    float2 extent;
    if (!compute_aabb_2dgs(T_row0, T_row1, T_row2, cutoff, center, extent)) {
        return;
    }
    // Low-pass filter floor on radius (matches forward.cu line 230).
    float radius = ceil(max(max(extent.x, extent.y), cutoff * FILTER_SIZE_2DGS));

    // Tile binding (same as 3DGS path).
    uint2 tile_min, tile_max;
    get_tile_bbox(center, float2(radius, radius), (int3)tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        return;
    }

    // Write per-surfel state.
    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    write_packed_float2(xys, idx, center);
    aabb[idx * 2]     = radius;
    aabb[idx * 2 + 1] = radius;

    // Write T as 3 rows × 3 floats = 9 floats per gaussian.
    // Layout matches hbb1's transMats: [row0.x, row0.y, row0.z, row1.x, ..., row2.z]
    // so the rasterizer reads contiguous float3s as the two homogeneous planes.
    uint t_base = idx * 9;
    transMats[t_base + 0] = T_row0.x;
    transMats[t_base + 1] = T_row0.y;
    transMats[t_base + 2] = T_row0.z;
    transMats[t_base + 3] = T_row1.x;
    transMats[t_base + 4] = T_row1.y;
    transMats[t_base + 5] = T_row1.z;
    transMats[t_base + 6] = T_row2.x;
    transMats[t_base + 7] = T_row2.y;
    transMats[t_base + 8] = T_row2.z;

    // Pack world-space normal + raw opacity (matches hbb1's normal_opacity output).
    write_packed_float4(normal_opacity, idx, float4(normal_world, opacities_raw[idx]));

    // SH evaluation — identical to the 3DGS path (color model is unchanged in 2DGS).
    float3 viewdir = normalize(p_world - cam_pos);
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint dc_idx = num_channels * idx;
    uint rest_idx = (num_bases - 1) * num_channels * idx;
    uint idx_col = num_channels * idx;
    sh_coeffs_to_color(degrees_to_use, viewdir, &(features_dc[dc_idx]), &(features_rest[rest_idx]), &(colors[idx_col]));
}

// Adam update helper — applies one Adam step to a single element.
// Computes in registers, writes param/exp_avg/exp_avg_sq back to device memory.
inline void adam_update_element(
    device float& param, device float& ea, device float& eas,
    float grad, float step_size, float beta1, float beta2, float bc2_sqrt, float eps
) {
    float m = fma(beta1, ea, (1.0f - beta1) * grad);
    float v = fma(beta2, eas, (1.0f - beta2) * grad * grad);
    param -= step_size * m / (sqrt(v) / bc2_sqrt + eps);
    ea = m;
    eas = v;
}

// Packed Adam hyperparameters for SH groups (passed via setBytes)
struct SHAdamParams {
    float dc_step_size;
    float dc_bc2_sqrt;
    float rest_step_size;
    float rest_bc2_sqrt;
    float beta1;
    float beta2;
    float eps;
};


// ===== Tile-Local Sorting Kernels =====
// Pre-allocated per-tile bins: scatter directly to fixed-size bins, then sort in-place.
// Eliminates count→prefix_sum→scatter pipeline (3 dispatches + 3 barriers saved).

#define SORT_TG_SIZE 256
#define MAX_TILE_ELEMS 2048

// Scatter each gaussian's intersections directly into pre-allocated per-tile bins.
// Each tile gets MAX_TILE_ELEMS slots. Per-tile atomics track fill count.
kernel void scatter_to_prealloc_bins_kernel(
    constant uint& num_points               [[buffer(0)]],
    constant float* xys                     [[buffer(1)]],
    constant float* depths                  [[buffer(2)]],
    constant int* radii                     [[buffer(3)]],
    constant float* aabb                    [[buffer(4)]],
    constant uint3& tile_bounds             [[buffer(5)]],
    device atomic_uint* scatter_counters    [[buffer(6)]],
    device uint64_t* prealloc_bins          [[buffer(7)]],
    device atomic_uint* overflow_flag       [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) return;
    if (radii[idx] <= 0) return;

    float2 center = read_packed_float2(xys, idx);
    uint2 tile_min, tile_max;
    get_tile_bbox(center, read_packed_float2(aabb, idx), (int3)tile_bounds, tile_min, tile_max);

    uint depth_bits = as_type<uint>(depths[idx]);

    for (uint i = tile_min.y; i < tile_max.y; i++) {
        for (uint j = tile_min.x; j < tile_max.x; j++) {
            uint tile_id = i * tile_bounds.x + j;
            uint pos = atomic_fetch_add_explicit(&scatter_counters[tile_id], 1u, memory_order_relaxed);
            if (pos >= MAX_TILE_ELEMS) {
                // Clamp counter so prefix_sum sees at most MAX_TILE_ELEMS
                atomic_store_explicit(&scatter_counters[tile_id], MAX_TILE_ELEMS, memory_order_relaxed);
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                continue;
            }
            prealloc_bins[(uint64_t)tile_id * MAX_TILE_ELEMS + pos] = ((uint64_t)depth_bits << 32) | (uint64_t)idx;
        }
    }
}

// Bitonic sort per tile in shared memory. Reads from pre-allocated bins.
// Writes sorted packed data to contiguous output (using tile_offsets from prefix sum).
// Also writes tile_bins for the rasterizer.

// ===== 2DGS bitonic sort + pack (Phase 2b.3.1, step 4) =====
// Same bitonic-sort scaffolding as bitonic_sort_per_tile_kernel — only the
// fused pack step differs. Reads per-gaussian transMat (9 floats) and
// normal_opacity (float4 with raw logit-space opacity in .w as written by
// project_and_sh_forward_2dgs_kernel), applies sigmoid to opacity, and writes
// the per-tile packed buffers consumed by nd_rasterize_forward_2dgs_kernel.
//
// Dead code until the host dispatcher's encode_prefix_map learns to call this
// variant.
kernel void bitonic_sort_per_tile_2dgs_kernel(
    constant int* tile_offsets              [[buffer(0)]],
    constant int* tile_counts_in            [[buffer(1)]],
    constant uint64_t* prealloc_bins        [[buffer(2)]],
    device int32_t* gaussian_ids_out        [[buffer(3)]],
    constant uint& num_tiles                [[buffer(4)]],
    // Pack inputs (per-gaussian).
    constant float* xys                     [[buffer(5)]],   // float2
    constant float* transMats               [[buffer(6)]],   // 9 floats
    constant float* normal_opacity_in       [[buffer(7)]],   // float4: (n.xyz, raw opacity logit)
    constant float* colors                  [[buffer(8)]],   // float3 (raw SH output)
    // Pack outputs (per sorted-gaussian).
    device float* packed_xy                 [[buffer(9)]],   // float2
    device float* packed_transmat           [[buffer(10)]],  // 9 floats
    device float* packed_normal_opac        [[buffer(11)]],  // float4 — .w is sigmoid(opac)
    device float* packed_rgb                [[buffer(12)]],  // float3
    device int* tile_bins                   [[buffer(13)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tg_id >= num_tiles) return;

    int count_raw = tile_counts_in[tg_id];
    int count = min(count_raw, MAX_TILE_ELEMS);
    int end = tile_offsets[tg_id];
    int start = end - count;

    if (tid == 0) {
        write_packed_int2(tile_bins, tg_id, int2(start, end));
    }
    if (count == 0) return;

    int n = 1;
    while (n < count) n <<= 1;

    threadgroup uint64_t data[MAX_TILE_ELEMS];
    uint64_t bin_base = (uint64_t)tg_id * MAX_TILE_ELEMS;
    for (int i = (int)tid; i < n; i += SORT_TG_SIZE) {
        data[i] = (i < count) ? prealloc_bins[bin_base + i] : 0xFFFFFFFFFFFFFFFFULL;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort (identical to 3DGS variant — uint64 keys with depth in upper bits).
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = (int)tid; i < (n >> 1); i += SORT_TG_SIZE) {
                int pos = 2 * i - (i & (j - 1));
                int partner = pos ^ j;
                bool ascending = ((pos & k) == 0);
                uint64_t a = data[pos];
                uint64_t b = data[partner];
                if ((a > b) == ascending) {
                    data[pos] = b;
                    data[partner] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 2DGS pack: copy xy, T (9 floats), normal+sigmoid(opacity), RGB
    for (int i = (int)tid; i < count; i += SORT_TG_SIZE) {
        int32_t g_id = (int32_t)(data[i] & 0xFFFFFFFF);
        int global_idx = start + i;
        gaussian_ids_out[global_idx] = g_id;

        // xy → float2
        write_packed_float2(packed_xy, global_idx, read_packed_float2(xys, g_id));

        // 9-float transmat — straight copy of the 3 rows
        uint src = (uint)g_id * 9;
        uint dst = (uint)global_idx * 9;
        for (int kk = 0; kk < 9; kk++) {
            packed_transmat[dst + kk] = transMats[src + kk];
        }

        // normal_opacity: read (n.xyz, raw_opac_logit), write (n.xyz, sigmoid(raw))
        float4 no = read_packed_float4(normal_opacity_in, g_id);
        float opac = 1.f / (1.f + exp(-no.w));
        write_packed_float4(packed_normal_opac, global_idx, float4(no.x, no.y, no.z, opac));

        // RGB straight copy (raw SH, matches 3DGS path)
        write_packed_float3(packed_rgb, global_idx, read_packed_float3(colors, g_id));
    }
}

// ===== 2DGS rasterize backward (M2.2) =====
// Per-pixel reverse traversal of the same surfels that contributed in the
// forward pass. Re-derives alpha (same arithmetic as forward), then computes
// VJPs for transMat / mean2D / normal3D / opacity / RGB and atomically
// scatters them to per-gaussian gradient buffers.
//
// Port of hbb1/2d-gaussian-splatting backward.cu:renderCUDA (lines 144-446).
// MSL 3.0 atomic_fetch_add_explicit on `device atomic<float>*` provides the
// equivalent of CUDA atomicAdd. Apple GPU family 7+ (M1 / A14+) — well within
// our target (iPad Pro M2, iPhone 14 Pro A16).
//
// Dead code until M2.5 wires the dispatcher. Loaded eagerly at startup so any
// kernel-creation error surfaces at init rather than first backward call.
kernel void nd_rasterize_backward_2dgs_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant int* tile_bins,                          // int2 (start, end) per tile
    constant int* gaussian_ids,                       // sorted point list — maps sorted_idx -> global gaussian id
    constant float* packed_xy,                        // float2 per sorted-surfel
    constant float* packed_normal_opac,               // float4 per sorted-surfel (n.xyz, sigmoid(opac))
    constant float* packed_transmat,                  // 9 floats per sorted-surfel
    constant float* packed_rgb,                       // float3 per sorted-surfel (raw SH, pre +0.5 clamp)
    constant float* final_Ts,                         // float3 per pixel (T_final, M1, M2)
    constant int* final_index,                        // int2 per pixel (last_contributor, median_contributor)
    constant float* background,                       // float3
    // upstream gradients
    constant float* dL_dout_img,                      // float3 per pixel
    constant float* dL_dout_depth,                    // float per pixel
    constant float* dL_dout_alpha,                    // float per pixel
    constant float* dL_dout_normal,                   // float3 per pixel
    constant float* dL_dout_median_depth,             // float per pixel
    constant float* dL_dout_distortion,               // float per pixel
    // outputs (atomic accumulators, sized per-gaussian)
    device atomic<float>* dL_dtransMat,               // [N, 9]
    device atomic<float>* dL_dmean2D,                 // [N, 2]
    device atomic<float>* dL_dnormal3D,               // [N, 3]
    device atomic<float>* dL_dopacity,                // [N, 1]
    device atomic<float>* dL_dcolors,                 // [N, 3]
    constant uint2& blockDim,
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]]
) {
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * (int)img_size.x + j;

    const bool inside = (i < (int)img_size.y && j < (int)img_size.x);

    int2 range = read_packed_int2(tile_bins, tile_id);
    const int rounds = (range.y - range.x + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;
    int toDo = range.y - range.x;

    // Threadgroup-shared per-surfel state for one reverse-batch.
    threadgroup int   collected_id[RAST_BLOCK_SIZE];
    threadgroup float2 collected_xy[RAST_BLOCK_SIZE];
    threadgroup float4 collected_normal_opac[RAST_BLOCK_SIZE];
    threadgroup float3 collected_Tu[RAST_BLOCK_SIZE];
    threadgroup float3 collected_Tv[RAST_BLOCK_SIZE];
    threadgroup float3 collected_Tw[RAST_BLOCK_SIZE];
    threadgroup float3 collected_rgb[RAST_BLOCK_SIZE];

    // Per-pixel forward outputs needed to re-derive gradients.
    float3 T_M1_M2     = inside ? float3(final_Ts[3*pix_id+0], final_Ts[3*pix_id+1], final_Ts[3*pix_id+2]) : float3(0);
    const float T_final = T_M1_M2.x;
    const float final_D = T_M1_M2.y;
    const float final_D2 = T_M1_M2.z;
    const float final_A = 1.0f - T_final;

    int2 contribs = inside ? int2(final_index[2*pix_id+0], final_index[2*pix_id+1]) : int2(-1, -1);
    const int last_contributor   = contribs.x;
    const int median_contributor = contribs.y;

    // Per-pixel upstream gradients.
    float3 dL_dpixel = {0,0,0};
    float dL_ddepth = 0.f;
    float dL_daccum = 0.f;
    float3 dL_dnormal2D = {0,0,0};
    float dL_dmedian_depth = 0.f;
    float dL_dreg = 0.f;
    float3 bg = float3(background[0], background[1], background[2]);
    if (inside) {
        dL_dpixel = float3(dL_dout_img[3*pix_id+0], dL_dout_img[3*pix_id+1], dL_dout_img[3*pix_id+2]);
        dL_ddepth = dL_dout_depth[pix_id];
        dL_daccum = dL_dout_alpha[pix_id];
        dL_dnormal2D = float3(dL_dout_normal[3*pix_id+0], dL_dout_normal[3*pix_id+1], dL_dout_normal[3*pix_id+2]);
        dL_dmedian_depth = dL_dout_median_depth[pix_id];
        dL_dreg = dL_dout_distortion[pix_id];
    }

    // Running state for the reverse traversal.
    float T = T_final;
    float3 accum_rec = {0,0,0};
    float3 last_color = {0,0,0};
    float last_alpha = 0.f;
    float accum_depth_rec = 0.f;
    float last_depth = 0.f;
    float accum_alpha_rec = 0.f;
    float3 accum_normal_rec = {0,0,0};
    float3 last_normal = {0,0,0};
    float last_dL_dT = 0.f;

    int contributor = toDo;
    bool done = !inside;

    for (int r = 0; r < rounds; ++r, toDo -= RAST_BLOCK_SIZE) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load batch in REVERSE order — last-contributor at index 0.
        int progress = r * RAST_BLOCK_SIZE + (int)tr;
        if (range.x + progress < range.y) {
            int sorted_idx = range.y - progress - 1;
            collected_id[tr]            = gaussian_ids[sorted_idx];
            collected_xy[tr]            = read_packed_float2(packed_xy, sorted_idx);
            collected_normal_opac[tr]   = read_packed_float4(packed_normal_opac, sorted_idx);
            uint t_base = (uint)sorted_idx * 9;
            collected_Tu[tr] = float3(packed_transmat[t_base+0], packed_transmat[t_base+1], packed_transmat[t_base+2]);
            collected_Tv[tr] = float3(packed_transmat[t_base+3], packed_transmat[t_base+4], packed_transmat[t_base+5]);
            collected_Tw[tr] = float3(packed_transmat[t_base+6], packed_transmat[t_base+7], packed_transmat[t_base+8]);
            collected_rgb[tr] = read_packed_float3(packed_rgb, sorted_idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (done) continue;

        int batch_size = min(RAST_BLOCK_SIZE, toDo);
        for (int j2 = 0; j2 < batch_size; ++j2) {
            // contributor counts down through the same surfels the forward
            // visited, in reverse order. Surfels past last_contributor never
            // contributed in the forward pass — skip them in backward too.
            contributor--;
            if (contributor >= last_contributor) continue;

            // Recompute ray-plane intersection (same as forward).
            const float3 Tu = collected_Tu[j2];
            const float3 Tv = collected_Tv[j2];
            const float3 Tw = collected_Tw[j2];
            const float3 k = px * Tw - Tu;
            const float3 l = py * Tw - Tv;
            const float3 p = cross(k, l);
            if (p.z == 0.0f) continue;
            const float2 s = float2(p.x / p.z, p.y / p.z);
            const float rho3d = s.x * s.x + s.y * s.y;
            const float2 xy = collected_xy[j2];
            const float2 d_screen = float2(xy.x - px, xy.y - py);
            const float rho2d = FILTER_INV_SQ_2DGS * (d_screen.x * d_screen.x + d_screen.y * d_screen.y);
            const float rho = min(rho3d, rho2d);

            // Per-pixel depth at ray-plane intersection.
            const float c_d = s.x * Tw.x + s.y * Tw.y + Tw.z;
            if (c_d < NEAR_N_2DGS) continue;

            const float power = -0.5f * rho;
            if (power > 0.0f) continue;

            const float4 nor_o = collected_normal_opac[j2];
            const float3 normal = float3(nor_o.x, nor_o.y, nor_o.z);
            const float opa = nor_o.w;
            const float G = exp(power);
            const float alpha = min(0.99f, opa * G);
            if (alpha < 1.0f / 255.0f) continue;

            // Recover prior T (T_before_this_surfel = T_after / (1 - alpha)).
            T = T / (1.0f - alpha);
            const float dchannel_dcolor = alpha * T;
            const float w = alpha * T;

            float dL_dalpha = 0.0f;
            const int global_id = collected_id[j2];

            // Propagate gradients on color. Forward applies max(raw + 0.5, 0)
            // when loading packed_rgb into the rasterizer's batch — so the
            // value that actually contributes to pix_color is the post-clamp.
            // Backward replays that: use the post-clamp value in the
            // (c - accum_rec) chain, and mask the dL/d(raw) gradient with
            // the clamp's derivative.
            for (int ch = 0; ch < 3; ch++) {
                const float c_raw  = collected_rgb[j2][ch];
                const float c_used = max(c_raw + 0.5f, 0.0f);
                const float dc_used_dc_raw = (c_raw + 0.5f > 0.0f) ? 1.0f : 0.0f;
                accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                last_color[ch] = c_used;
                const float dL_dch = dL_dpixel[ch];
                dL_dalpha += (c_used - accum_rec[ch]) * dL_dch;
                atomic_fetch_add_explicit(&dL_dcolors[3 * global_id + ch],
                                          dchannel_dcolor * dL_dch * dc_used_dc_raw,
                                          memory_order_relaxed);
            }

            // Aux-output gradients: median depth, distortion, depth, alpha, normal.
            float dL_dz = 0.0f;
            float dL_dweight = 0.0f;

            const float m_d = FAR_N_2DGS / (FAR_N_2DGS - NEAR_N_2DGS) * (1.0f - NEAR_N_2DGS / c_d);
            const float dmd_dd = (FAR_N_2DGS * NEAR_N_2DGS) / ((FAR_N_2DGS - NEAR_N_2DGS) * c_d * c_d);
            if (contributor == median_contributor - 1) {
                dL_dz += dL_dmedian_depth;
            }
            // Distortion gradient: weight gradient + dL_dz contribution.
            // DETACH_WEIGHT=0 path — match backward.cu line 356.
            dL_dweight += (final_D2 + m_d * m_d * final_A - 2.0f * m_d * final_D) * dL_dreg;
            dL_dalpha += dL_dweight - last_dL_dT;
            last_dL_dT = dL_dweight * alpha + (1.0f - alpha) * last_dL_dT;
            const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
            dL_dz += dL_dmd * dmd_dd;

            // Ray-splat depth → alpha-weighted depth output.
            accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
            last_depth = c_d;
            dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;

            // Accumulated alpha output.
            accum_alpha_rec = last_alpha * 1.0f + (1.f - last_alpha) * accum_alpha_rec;
            dL_dalpha += (1.0f - accum_alpha_rec) * dL_daccum;

            // Normal gradients.
            for (int ch = 0; ch < 3; ch++) {
                accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
                last_normal[ch] = normal[ch];
                dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
                atomic_fetch_add_explicit(&dL_dnormal3D[3 * global_id + ch], alpha * T * dL_dnormal2D[ch], memory_order_relaxed);
            }

            dL_dalpha *= T;
            last_alpha = alpha;

            // Background contribution to alpha gradient.
            const float bg_dot = bg.x * dL_dpixel.x + bg.y * dL_dpixel.y + bg.z * dL_dpixel.z;
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot;

            // Helpful temporary: gradient on G (the un-clamped Gaussian factor).
            const float dL_dG = nor_o.w * dL_dalpha;
            dL_dz += alpha * T * dL_ddepth;

            // Branch on which rho dominated the per-pixel evaluation.
            if (rho3d <= rho2d) {
                // 3D path: gradient flows through ray-plane intersection (s) into transMat.
                const float2 dL_ds = float2(
                    dL_dG * -G * s.x + dL_dz * Tw.x,
                    dL_dG * -G * s.y + dL_dz * Tw.y);
                const float3 dz_dTw = float3(s.x, s.y, 1.0f);
                const float dsx_pz = dL_ds.x / p.z;
                const float dsy_pz = dL_ds.y / p.z;
                const float3 dL_dp = float3(dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y));
                const float3 dL_dk = cross(l, dL_dp);
                const float3 dL_dl = cross(dL_dp, k);
                const float3 dL_dTu = float3(-dL_dk.x, -dL_dk.y, -dL_dk.z);
                const float3 dL_dTv = float3(-dL_dl.x, -dL_dl.y, -dL_dl.z);
                const float3 dL_dTw = float3(
                    px * dL_dk.x + py * dL_dl.x + dL_dz * dz_dTw.x,
                    px * dL_dk.y + py * dL_dl.y + dL_dz * dz_dTw.y,
                    px * dL_dk.z + py * dL_dl.z + dL_dz * dz_dTw.z);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+0], dL_dTu.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+1], dL_dTu.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+2], dL_dTu.z, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+3], dL_dTv.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+4], dL_dTv.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+5], dL_dTv.z, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+6], dL_dTw.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+7], dL_dTw.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+8], dL_dTw.z, memory_order_relaxed);
            } else {
                // 2D path: low-pass filter clamped — gradient flows through screen-space distance.
                const float dG_ddelx = -G * FILTER_INV_SQ_2DGS * d_screen.x;
                const float dG_ddely = -G * FILTER_INV_SQ_2DGS * d_screen.y;
                atomic_fetch_add_explicit(&dL_dmean2D[2*global_id+0], dL_dG * dG_ddelx, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dmean2D[2*global_id+1], dL_dG * dG_ddely, memory_order_relaxed);
                // Depth gradient still routed through transMat row 2 (Tw).
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+6], s.x * dL_dz, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+7], s.y * dL_dz, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_dtransMat[9*global_id+8], dL_dz, memory_order_relaxed);
            }

            // Opacity gradient. The forward applies sigmoid to the raw logit
            // when packing (line 1573 above), so the rasterizer sees
            // opa = sigmoid(raw). To produce dL/d(raw) — the gradient Adam
            // applies to `opacities` — we compose the sigmoid derivative
            // d(sigmoid(raw))/d(raw) = opa * (1 - opa). Without this, opacity
            // gradients are wrong-magnitude (and effectively zero near
            // saturation), so splats with opacity≈0 or ≈1 cannot adjust.
            const float dsig = opa * (1.0f - opa);
            atomic_fetch_add_explicit(&dL_dopacity[global_id], G * dL_dalpha * dsig, memory_order_relaxed);
        }
    }
}

// ===== 2DGS project + SH backward (M2.3) =====
// One thread per gaussian. Reads the atomic-accumulated gradients
// (dL_dtransMat, dL_dnormal3D, dL_dcolors) from rasterize_backward_2dgs, plus
// the forward parameters (means3D, scales, quats, viewmat, projmat), and
// produces the final per-gaussian gradients:
//
//   dL_dmean3D        [N, 3]
//   dL_dscale         [N, 2]
//   dL_dquat          [N, 4]
//   dL_dfeatures_dc   [N, 3]
//   dL_dfeatures_rest [N, frBases, 3]
//   dL_dmean2D        [N, 2]   (OVERWRITTEN here with the NDC-space formula
//                              that densify reads; the rasterizer's atomic
//                              accumulation into mean2D is discarded.)
//
// Derivation: this kernel re-derives the chain rule for the row-major T
// construction in compute_transmat_2dgs (lines 226-295). The reference's
// backward.cu:compute_transmat_aabb assumes glm column-major storage, so
// it can't be transliterated — instead this is a direct VJP through the
// (R, scale, projmat, halfW/H, cx/cy) → T formula in our forward.
//
// Dead code until M2.5 wires dispatch.
kernel void project_and_sh_backward_2dgs_kernel(
    constant int& num_points,
    constant float* means3D,            // float3 per gaussian
    constant float* scales,             // float2 per gaussian (log-space)
    constant float& glob_scale,
    constant float* quats,              // float4 per gaussian
    constant float* viewmat,            // 4x4 row-major
    constant float* projmat,            // 4x4 row-major (world → clip)
    constant float4& intrins,           // (fx, fy, cx, cy)
    constant uint2& img_size,
    constant int* radii,                // per-gaussian (forward output; 0 = culled)
    // Upstream gradients
    constant float* dL_dtransMat,       // [N, 9]
    constant float* dL_dnormal3D,       // [N, 3]
    constant float* dL_dcolors,         // [N, 3] (gradient on raw SH output, post mask)
    // SH inputs
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float3& cam_pos,
    // SH coefficient buffers (needed for dL/d(viewdir) → dL/d(mean3D) chain;
    // reference computeColorFromSH backward at backward.cu lines 36-138).
    constant float* features_rest,      // [N, frBases, 3] — non-DC SH bases
    constant uint& num_bases_total,     // = num_sh_bases(degree)
    // Outputs
    device float* dL_dmean3D,           // [N, 3]
    device float* dL_dscale,            // [N, 2]
    device float* dL_dquat,             // [N, 4]
    device float* dL_dmean2D,           // [N, 2]  (OVERWRITE for densify hack)
    device float* dL_dfeatures_dc,      // [N, 3]
    device float* dL_dfeatures_rest,    // [N, frBases, 3]
    // Densify gradient-stat accumulators (read-modify-write; reset by host every
    // refineEvery iters after densify runs). Without these populated, densify
    // sees zeros and never splits / dups → splat count stays at random-init
    // count and training plateaus. Bug discovered during dummyhead 30k smoke.
    device float* xys_grad_norm,        // [N]   — Σ |dL/dmean2D| (densify hack version)
    device float* vis_counts,           // [N]   — count of iters this gaussian was visible
    device float* max_2d_size,          // [N]   — max screen-space radius (in normalized units)
    constant float& inv_max_dim,        // 1 / max(W, H) so radii are scale-invariant
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= (uint)num_points) return;

    // Culled gaussians: zero their per-gaussian gradients and bail.
    if (radii[idx] <= 0) {
        for (int k = 0; k < 3; k++) dL_dmean3D[3*idx+k] = 0.0f;
        for (int k = 0; k < 2; k++) dL_dscale[2*idx+k] = 0.0f;
        for (int k = 0; k < 4; k++) dL_dquat[4*idx+k] = 0.0f;
        for (int k = 0; k < 2; k++) dL_dmean2D[2*idx+k] = 0.0f;
        for (int k = 0; k < 3; k++) dL_dfeatures_dc[3*idx+k] = 0.0f;
        // dL_dfeatures_rest is zero-init'd by host blit before this dispatch.
        return;
    }

    // === Recompute forward state ===
    float3 p_orig = read_packed_float3(means3D, idx);
    float2 s_log = read_packed_float2(scales, idx);
    float2 scale_2d = exp(s_log);
    float4 quat = read_packed_float4(quats, idx);
    float3x3 R = quat_to_rotmat(quat);
    float sx = scale_2d.x * glob_scale;
    float sy = scale_2d.y * glob_scale;
    float3 tu_world = R[0] * sx;
    float3 tv_world = R[1] * sy;
    float3 normal_world = R[2];

    // tu_clip / tv_clip / p_clip from projmat (row-major).
    float4 tu_clip = float4(
        projmat[0]*tu_world.x + projmat[1]*tu_world.y + projmat[2]*tu_world.z,
        projmat[4]*tu_world.x + projmat[5]*tu_world.y + projmat[6]*tu_world.z,
        projmat[8]*tu_world.x + projmat[9]*tu_world.y + projmat[10]*tu_world.z,
        projmat[12]*tu_world.x + projmat[13]*tu_world.y + projmat[14]*tu_world.z);
    float4 tv_clip = float4(
        projmat[0]*tv_world.x + projmat[1]*tv_world.y + projmat[2]*tv_world.z,
        projmat[4]*tv_world.x + projmat[5]*tv_world.y + projmat[6]*tv_world.z,
        projmat[8]*tv_world.x + projmat[9]*tv_world.y + projmat[10]*tv_world.z,
        projmat[12]*tv_world.x + projmat[13]*tv_world.y + projmat[14]*tv_world.z);
    float4 p_clip = float4(
        projmat[0]*p_orig.x + projmat[1]*p_orig.y + projmat[2]*p_orig.z + projmat[3],
        projmat[4]*p_orig.x + projmat[5]*p_orig.y + projmat[6]*p_orig.z + projmat[7],
        projmat[8]*p_orig.x + projmat[9]*p_orig.y + projmat[10]*p_orig.z + projmat[11],
        projmat[12]*p_orig.x + projmat[13]*p_orig.y + projmat[14]*p_orig.z + projmat[15]);
    float W = (float)img_size.x;
    float H = (float)img_size.y;
    float halfW = 0.5f * W;
    float halfH = 0.5f * H;
    float bx = intrins.z - 0.5f;
    float by = intrins.w - 0.5f;

    // === Chain rule ===
    // dL_d(T_row_r[c]) is at dL_dtransMat[9*idx + 3*r + c].
    // T_row0 = (col_u.x, col_v.x, col_o.x),
    // T_row1 = (col_u.y, col_v.y, col_o.y),
    // T_row2 = (col_u.z, col_v.z, col_o.z).
    // So dL_dcol_u = (dL_dT_row0[0], dL_dT_row1[0], dL_dT_row2[0])
    //             = (dL_dT[0], dL_dT[3], dL_dT[6])  per-gaussian (without idx*9).
    uint base = idx * 9;
    float3 dL_dcol_u = float3(dL_dtransMat[base+0], dL_dtransMat[base+3], dL_dtransMat[base+6]);
    float3 dL_dcol_v = float3(dL_dtransMat[base+1], dL_dtransMat[base+4], dL_dtransMat[base+7]);
    float3 dL_dcol_o = float3(dL_dtransMat[base+2], dL_dtransMat[base+5], dL_dtransMat[base+8]);

    // col_u = (halfW*tu_clip.x + bx*tu_clip.w,
    //         halfH*tu_clip.y + by*tu_clip.w,
    //         tu_clip.w)
    // tu_clip.z does not appear in col_u; dL_dtu_clip.z = 0.
    float4 dL_dtu_clip;
    dL_dtu_clip.x = halfW * dL_dcol_u.x;
    dL_dtu_clip.y = halfH * dL_dcol_u.y;
    dL_dtu_clip.z = 0.0f;
    dL_dtu_clip.w = bx * dL_dcol_u.x + by * dL_dcol_u.y + dL_dcol_u.z;
    float4 dL_dtv_clip;
    dL_dtv_clip.x = halfW * dL_dcol_v.x;
    dL_dtv_clip.y = halfH * dL_dcol_v.y;
    dL_dtv_clip.z = 0.0f;
    dL_dtv_clip.w = bx * dL_dcol_v.x + by * dL_dcol_v.y + dL_dcol_v.z;
    float4 dL_dp_clip;
    dL_dp_clip.x = halfW * dL_dcol_o.x;
    dL_dp_clip.y = halfH * dL_dcol_o.y;
    dL_dp_clip.z = 0.0f;
    dL_dp_clip.w = bx * dL_dcol_o.x + by * dL_dcol_o.y + dL_dcol_o.z;

    // tu_clip[i] = sum_{j=0..2} projmat[4i+j] * tu_world[j]   (no w input).
    // dL_dtu_world[j] = sum_i projmat[4i+j] * dL_dtu_clip[i].
    float3 dL_dtu_world;
    dL_dtu_world.x = projmat[0]*dL_dtu_clip.x + projmat[4]*dL_dtu_clip.y + projmat[8]*dL_dtu_clip.z + projmat[12]*dL_dtu_clip.w;
    dL_dtu_world.y = projmat[1]*dL_dtu_clip.x + projmat[5]*dL_dtu_clip.y + projmat[9]*dL_dtu_clip.z + projmat[13]*dL_dtu_clip.w;
    dL_dtu_world.z = projmat[2]*dL_dtu_clip.x + projmat[6]*dL_dtu_clip.y + projmat[10]*dL_dtu_clip.z + projmat[14]*dL_dtu_clip.w;
    float3 dL_dtv_world;
    dL_dtv_world.x = projmat[0]*dL_dtv_clip.x + projmat[4]*dL_dtv_clip.y + projmat[8]*dL_dtv_clip.z + projmat[12]*dL_dtv_clip.w;
    dL_dtv_world.y = projmat[1]*dL_dtv_clip.x + projmat[5]*dL_dtv_clip.y + projmat[9]*dL_dtv_clip.z + projmat[13]*dL_dtv_clip.w;
    dL_dtv_world.z = projmat[2]*dL_dtv_clip.x + projmat[6]*dL_dtv_clip.y + projmat[10]*dL_dtv_clip.z + projmat[14]*dL_dtv_clip.w;
    float3 dL_dp_orig;
    dL_dp_orig.x = projmat[0]*dL_dp_clip.x + projmat[4]*dL_dp_clip.y + projmat[8]*dL_dp_clip.z + projmat[12]*dL_dp_clip.w;
    dL_dp_orig.y = projmat[1]*dL_dp_clip.x + projmat[5]*dL_dp_clip.y + projmat[9]*dL_dp_clip.z + projmat[13]*dL_dp_clip.w;
    dL_dp_orig.z = projmat[2]*dL_dp_clip.x + projmat[6]*dL_dp_clip.y + projmat[10]*dL_dp_clip.z + projmat[14]*dL_dp_clip.w;

    // tu_world = R[0] * sx  →  dL_dR[0] = dL_dtu_world * sx,
    //                          dL_dsx   = dot(dL_dtu_world, R[0])
    // sx = scale_2d.x * glob_scale; scale_2d.x = exp(s_log.x); dL_d(s_log.x) = dL_dsx * scale_2d.x * glob_scale.
    float dL_dsx = dot(dL_dtu_world, R[0]);
    float dL_dsy = dot(dL_dtv_world, R[1]);
    float dL_dslog_x = dL_dsx * scale_2d.x * glob_scale;
    float dL_dslog_y = dL_dsy * scale_2d.y * glob_scale;

    // Gradient on R columns. R[2] (normal) gets dL_dnormal3D directly.
    float3 dL_dR_col0 = dL_dtu_world * sx;
    float3 dL_dR_col1 = dL_dtv_world * sy;
    float3 dL_dR_col2 = read_packed_float3(dL_dnormal3D, idx);

    // quat VJP: pack the 3 column gradients into a float3x3 and call the helper.
    float3x3 dL_dR = float3x3(dL_dR_col0, dL_dR_col1, dL_dR_col2);
    float4 dL_dquat_v = quat_to_rotmat_vjp(quat, dL_dR);

    // Write per-gaussian outputs (no atomic — one thread owns one gaussian).
    dL_dmean3D[3*idx+0] = dL_dp_orig.x;
    dL_dmean3D[3*idx+1] = dL_dp_orig.y;
    dL_dmean3D[3*idx+2] = dL_dp_orig.z;
    dL_dscale[2*idx+0] = dL_dslog_x;
    dL_dscale[2*idx+1] = dL_dslog_y;
    dL_dquat[4*idx+0] = dL_dquat_v.x;
    dL_dquat[4*idx+1] = dL_dquat_v.y;
    dL_dquat[4*idx+2] = dL_dquat_v.z;
    dL_dquat[4*idx+3] = dL_dquat_v.w;

    // SH chain rule. viewdir = normalize(p_world - cam_pos). dL_dcolors is the
    // gradient on raw SH eval (already masked by the +0.5/max clamp in the
    // rasterize backward).
    float3 dir_unnorm = p_orig - cam_pos;
    float3 viewdir = normalize(dir_unnorm);
    uint num_bases = num_sh_bases(degree);
    uint dc_idx = 3 * idx;
    uint rest_idx = idx * (num_bases - 1) * 3;
    sh_coeffs_to_color_vjp(
        degrees_to_use, viewdir,
        &dL_dcolors[3 * idx],
        &dL_dfeatures_dc[dc_idx],
        &dL_dfeatures_rest[rest_idx]);

    // Bug 2 fix: dL/d(viewdir) → dL/d(p_orig). For SH degree ≥ 1, color depends
    // on viewdir = normalize(p_orig - cam_pos). The reference accounts for this
    // (backward.cu lines 127-138); without it, mean3D gets a biased gradient
    // and trained gaussians don't converge to the correct 3D positions even
    // though the photometric loss matches.
    if (degrees_to_use >= 1) {
        float3 dL_ddir = sh_color_dL_ddir(
            degrees_to_use, viewdir,
            &features_rest[rest_idx],
            &dL_dcolors[3 * idx]);
        float3 dL_dp_orig_from_dir = dnormvdv(dir_unnorm, dL_ddir);
        dL_dmean3D[3*idx+0] += dL_dp_orig_from_dir.x;
        dL_dmean3D[3*idx+1] += dL_dp_orig_from_dir.y;
        dL_dmean3D[3*idx+2] += dL_dp_orig_from_dir.z;
    }

    // Densify proxy. Reference (column-major) uses slots 2 and 5 = col_u.z, col_v.z.
    // In our row-major storage those parameters live at slots 6 and 7 (= T_row2.x,
    // T_row2.y = dL/d(col_u.z), dL/d(col_v.z)).
    float depth_fwd = p_clip.w;  // = col_o.z
    float dmx = dL_dtransMat[base + 6] * depth_fwd * 0.5f * W;
    float dmy = dL_dtransMat[base + 7] * depth_fwd * 0.5f * H;
    dL_dmean2D[2*idx+0] = dmx;
    dL_dmean2D[2*idx+1] = dmy;
    xys_grad_norm[idx] += fabs(dmx) + fabs(dmy);
    vis_counts[idx]    += 1.0f;
    float r_norm = float(radii[idx]) * inv_max_dim;
    if (r_norm > max_2d_size[idx]) max_2d_size[idx] = r_norm;
}

// ===== TSDF fusion (Phase 2c.2) =====
// Per-voxel integration of one rendered depth map into a [Dz, Dy, Dx, 2] grid.
// Each voxel's last dim is (sdf, weight). One thread per voxel; one dispatch
// per camera. Dispatches are serial (no inter-camera concurrency), so the
// read-modify-write of (sdf, weight) needs no atomics.
//
// Math: project voxel center to camera image space using the same viewmat +
// pinhole intrinsics msplat's project_and_sh_forward_2dgs uses (CV convention,
// +Z forward in camera frame). Sample the rendered depth at that pixel. SDF
// is (sampled_mean_depth − voxel_view_z), positive = empty / in front of
// surface, negative = behind. Truncate at ±trunc. Accumulate weighted by
// the pixel's accumulated alpha (high-alpha pixels carry more depth info).
kernel void tsdf_integrate_kernel(
    constant uint3& dims,                // (Dx, Dy, Dz) voxels per axis
    constant float3& origin,             // world-space position of voxel (0,0,0) corner
    constant float& voxelSize,           // edge length, meters
    constant float* viewmat,             // 4x4 row-major (world → view, CV convention)
    constant float4& intrins,            // (fx, fy, cx, cy) at the rendered resolution
    constant uint2& imgSize,             // (W, H) of the depth/alpha buffers
    constant float& truncDist,           // truncation distance, meters
    constant float& alphaThresh,         // skip pixels below this accumulated alpha
    constant float* depthMap,            // [H, W] — out_depth (sum, not mean)
    constant float* alphaMap,            // [H, W] — out_alpha (= 1 - T_final)
    device float* grid,                  // [Dz, Dy, Dx, 2] in-place
    uint3 gp [[thread_position_in_grid]]
) {
    if (gp.x >= dims.x || gp.y >= dims.y || gp.z >= dims.z) return;

    // Voxel center in world coords.
    float3 wp = float3(
        origin.x + (float(gp.x) + 0.5f) * voxelSize,
        origin.y + (float(gp.y) + 0.5f) * voxelSize,
        origin.z + (float(gp.z) + 0.5f) * voxelSize);

    // World → view (row-major viewmat).
    float view_x = viewmat[0]*wp.x + viewmat[1]*wp.y + viewmat[2]*wp.z + viewmat[3];
    float view_y = viewmat[4]*wp.x + viewmat[5]*wp.y + viewmat[6]*wp.z + viewmat[7];
    float view_z = viewmat[8]*wp.x + viewmat[9]*wp.y + viewmat[10]*wp.z + viewmat[11];
    if (view_z <= 0.01f) return;  // behind / on the camera

    // Pinhole projection.
    float pix_x = view_x / view_z * intrins.x + intrins.z;
    float pix_y = view_y / view_z * intrins.y + intrins.w;
    int ix = (int)pix_x;
    int iy = (int)pix_y;
    if (ix < 0 || ix >= (int)imgSize.x || iy < 0 || iy >= (int)imgSize.y) return;

    int pid = iy * (int)imgSize.x + ix;
    float a = alphaMap[pid];
    if (a < alphaThresh) return;
    // depthMap is expected to be already-normalized (the model side either
    // passes median depth — surface depth at T=0.5 — or pre-divides expected
    // depth by alpha). Don't divide again here.
    float depth = depthMap[pid];
    if (!isfinite(depth) || depth <= 0.0f) return;

    // SDF in meters. Positive if voxel is between camera and surface.
    float sdf = depth - view_z;
    if (sdf < -truncDist) return;       // far behind surface, no info
    sdf = clamp(sdf, -truncDist, truncDist);

    // Voxel buffer index (matches host C++ shape [Dz, Dy, Dx, 2]).
    uint vid = ((gp.z * dims.y + gp.y) * dims.x + gp.x) * 2u;
    float old_sdf = grid[vid + 0];
    float old_w   = grid[vid + 1];
    float w       = a;
    float new_w   = old_w + w;
    grid[vid + 0] = (old_sdf * old_w + sdf * w) / new_w;
    grid[vid + 1] = new_w;
}

// ===== Marching Cubes (Phase 2c.3) =====
// Per-cell triangle emission from a TSDF voxel grid. Standard Lorensen/Cline
// algorithm with Paul Bourke's edge + triangle tables. One Metal thread per
// cell. Cells where any corner has weight ≤ 0 (unobserved by any camera) are
// skipped — avoids hallucinated surfaces in occluded regions.
//
// Output is non-indexed: each triangle gets 3 own vertices in `tris_out`
// (xyz per vertex). Atomic counter `tri_count` tracks how many triangles
// were emitted; host caps the output buffer size and reads back the count
// to know how many to write to the PLY.

// 256-entry edge table: bitmask of which of 12 cube edges are crossed for
// each of 2^8 corner sign configurations.
constant int MC_EDGE_TABLE[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

// 256 x 16 triangle table. Each row lists up to 5 triangles as edge indices
// (3 per triangle, -1 terminator after the last). Cells with no triangles
// have the row entirely -1.
// Source: http://paulbourke.net/geometry/polygonise/ (public domain).
constant int MC_TRI_TABLE[256 * 16] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1,
     3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1,
     3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1,
     3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1,
     9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1,
     1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,
     9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,
     2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1,
     8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1,
     9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,
     4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1,
     3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1,
     1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1,
     4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1,
     4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1,
     9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1,
     1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1,
     5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1,
     2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1,
     9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1,
     0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,
     2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1,
    10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,
     4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1,
     5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1,
     5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,
     9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1,
     0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1,
     1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1,
    10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1,
     8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1,
     2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,
     7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,
     9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1,
     2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1,
    11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1,
     9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1,
     5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1,
    11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1,
    11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,
     1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1,
     9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1,
     5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1,
     2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,
     0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,
     5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1,
     6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1,
     0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1,
     3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1,
     6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1,
     5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1,
     1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,
    10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1,
     6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,
     1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1,
     8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1,
     7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1,
     3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,
     5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1,
     0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1,
     9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1,
     8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1,
     5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1,
     0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1,
     6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1,
    10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,
    10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1,
     8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1,
     1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1,
     3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1,
     0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,
    10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1,
     3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1,
     6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1,
     9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1,
     8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1,
     3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1,
     6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1,
     0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1,
    10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1,
    10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1,
     1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1,
     2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1,
     7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1,
     7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1,
     2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1,
     1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1,
    11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1,
     8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1,
     0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1,
     7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,
    10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,
     2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,
     6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1,
     7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1,
     2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,
     1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1,
    10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1,
    10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1,
     0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1,
     7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1,
     6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1,
     8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1,
     9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1,
     6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1,
     1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1,
     4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1,
    10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1,
     8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,
     0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1,
     1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1,
     8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1,
    10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1,
     4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1,
    10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,
     5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,
    11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1,
     9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,
     6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1,
     7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1,
     3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1,
     7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1,
     9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1,
     3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1,
     6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1,
     9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1,
     1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1,
     4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1,
     7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1,
     6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1,
     3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1,
     0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1,
     6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1,
     1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1,
     0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1,
    11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1,
     6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1,
     5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1,
     9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1,
     1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1,
     1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1,
    10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1,
     0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1,
     5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1,
    10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1,
    11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1,
     9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1,
     7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1,
     2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,
     8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1,
     9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1,
     9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1,
     1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1,
     9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1,
     9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,
     5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1,
     0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1,
    10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1,
     2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1,
     0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1,
     0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1,
     9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1,
     5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1,
     3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1,
     5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1,
     8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1,
     0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1,
     9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1,
     0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1,
     1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1,
     3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1,
     4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1,
     9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1,
    11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1,
    11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1,
     2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1,
     9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1,
     3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1,
     1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1,
     4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1,
     4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1,
     0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1,
     3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1,
     3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1,
     0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1,
     9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1,
     1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
};

// Edge endpoint pairs: edge i connects MC_EDGE_VERT[i][0] and [1] of the cube.
constant int MC_EDGE_VERT[24] = {
    0, 1,   1, 2,   2, 3,   3, 0,
    4, 5,   5, 6,   6, 7,   7, 4,
    0, 4,   1, 5,   2, 6,   3, 7
};

// Cube corner offsets (matches Bourke's convention).
constant int3 MC_CORNER_OFFSET[8] = {
    int3(0,0,0), int3(1,0,0), int3(1,1,0), int3(0,1,0),
    int3(0,0,1), int3(1,0,1), int3(1,1,1), int3(0,1,1)
};

kernel void marching_cubes_kernel(
    constant uint3& dims,                   // grid voxel dims (Dx, Dy, Dz)
    constant float3& origin,                // world position of voxel (0,0,0)
    constant float& voxelSize,
    constant float* grid,                   // [Dz, Dy, Dx, 2] (sdf, weight) — read-only
    constant uint& maxTriangles,            // size of tris_out buffer in triangles
    device atomic_uint* tri_count,          // atomic counter, output
    device float* tris_out,                 // (Ntri × 9) world-space vertex coords
    uint3 gp [[thread_position_in_grid]]
) {
    // Cell (i, j, k) spans voxels (i, j, k) to (i+1, j+1, k+1). Bound check:
    // last cell index is dims - 2 (since the cell needs both endpoints).
    if (gp.x + 1 >= dims.x || gp.y + 1 >= dims.y || gp.z + 1 >= dims.z) return;

    // Read SDF and weight at all 8 corners.
    float sdf[8];
    float wts[8];
    for (int c = 0; c < 8; c++) {
        int3 off = MC_CORNER_OFFSET[c];
        uint idx = (((gp.z + off.z) * dims.y + (gp.y + off.y)) * dims.x + (gp.x + off.x)) * 2u;
        sdf[c] = grid[idx + 0];
        wts[c] = grid[idx + 1];
    }
    // Skip cells with any unobserved corner — avoids hallucinated surfaces.
    for (int c = 0; c < 8; c++) {
        if (wts[c] <= 0.0f) return;
    }

    // Build cube index from corner signs (1 = inside / negative SDF).
    int cubeIdx = 0;
    for (int c = 0; c < 8; c++) {
        if (sdf[c] < 0.0f) cubeIdx |= (1 << c);
    }
    int edgeMask = MC_EDGE_TABLE[cubeIdx];
    if (edgeMask == 0) return;

    // Linear-interpolate each crossed edge to get a vertex position.
    float3 edgeVerts[12];
    for (int e = 0; e < 12; e++) {
        if ((edgeMask & (1 << e)) == 0) continue;
        int v0 = MC_EDGE_VERT[e * 2 + 0];
        int v1 = MC_EDGE_VERT[e * 2 + 1];
        float s0 = sdf[v0];
        float s1 = sdf[v1];
        float t = (s1 == s0) ? 0.5f : (-s0 / (s1 - s0));
        int3 o0 = MC_CORNER_OFFSET[v0];
        int3 o1 = MC_CORNER_OFFSET[v1];
        float3 p0 = origin + voxelSize * float3(float(gp.x + o0.x) + 0.5f,
                                                 float(gp.y + o0.y) + 0.5f,
                                                 float(gp.z + o0.z) + 0.5f);
        float3 p1 = origin + voxelSize * float3(float(gp.x + o1.x) + 0.5f,
                                                 float(gp.y + o1.y) + 0.5f,
                                                 float(gp.z + o1.z) + 0.5f);
        edgeVerts[e] = mix(p0, p1, t);
    }

    // Emit triangles. Each row has up to 5 tris × 3 edge indices, -1 terminated.
    int rowBase = cubeIdx * 16;
    for (int t = 0; t < 16; t += 3) {
        int e0 = MC_TRI_TABLE[rowBase + t + 0];
        if (e0 < 0) break;
        int e1 = MC_TRI_TABLE[rowBase + t + 1];
        int e2 = MC_TRI_TABLE[rowBase + t + 2];
        float3 v0 = edgeVerts[e0];
        float3 v1 = edgeVerts[e1];
        float3 v2 = edgeVerts[e2];

        // Reserve a slot atomically.
        uint slot = atomic_fetch_add_explicit(tri_count, 1u, memory_order_relaxed);
        if (slot >= maxTriangles) return;   // output buffer full
        uint base = slot * 9u;
        tris_out[base + 0] = v0.x;
        tris_out[base + 1] = v0.y;
        tris_out[base + 2] = v0.z;
        tris_out[base + 3] = v1.x;
        tris_out[base + 4] = v1.y;
        tris_out[base + 5] = v1.z;
        tris_out[base + 6] = v2.x;
        tris_out[base + 7] = v2.y;
        tris_out[base + 8] = v2.z;
    }
}

// ===== 2DGS training losses (M2.4) =====
// Minimal first-pass: L1 image loss + depth-distortion regularizer. SSIM and
// normal-consistency are deferred — they add ~150 lines each and aren't
// strictly required to verify the backward chain end-to-end (M2.7 smoke
// just needs loss-down-over-iters from any well-conditioned loss).
//
// Produces upstream gradients for rasterize_backward_2dgs:
//   dL_dout_img       = sign(rendered - gt) / N        (per pixel × 3 channels)
//   dL_dout_distortion = lambda_dist / N               (uniform per pixel)
//   dL_dout_depth, dL_dout_alpha, dL_dout_normal,
//   dL_dout_median_depth                              = 0 (host blit zeroes them)
//
// Also computes the per-pixel loss contribution and atomically accumulates
// it into a scalar loss_sum for host-side reporting.

kernel void loss_l1_distortion_2dgs_kernel(
    constant uint2& img_size,
    constant float& inv_num_pixels,        // 1 / (H * W * 3) for L1 normalization
    constant float& lambda_l1,             // typically 0.8 (1 - ssim_weight)
    constant float& lambda_dist,           // depth distortion regularizer weight
    constant float& lambda_dssim,          // DSSIM weight (ssim_weight, e.g. 0.2)
    // Forward outputs (read).
    constant float* out_img,               // float3 per pixel — already +bg composited
    constant float* gt,                    // float3 per pixel
    constant float* out_distortion,        // float per pixel — needed for the regularizer loss value
    constant float* ssim_sum,              // [1] scalar — sum of ssim_map over (H*W*3), produced by ssim_compute_kernel
    // Gradient outputs (write — host blits to zero before this dispatch).
    device float* dL_dout_img,             // float3 per pixel
    device float* dL_dout_distortion,      // float per pixel
    // Scalar loss reduction.
    device atomic<float>* loss_sum,
    uint2 pix [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    if (pix.x >= W || pix.y >= H) return;
    uint pix_id = pix.y * W + pix.x;

    float local_loss = 0.0f;

    // L1 image loss per channel.
    for (int ch = 0; ch < 3; ch++) {
        float r = out_img[3 * pix_id + ch];
        float g = gt[3 * pix_id + ch];
        float diff = r - g;
        local_loss += lambda_l1 * fabs(diff) * inv_num_pixels;
        // dL/d(rendered) = lambda * sign(diff) * inv_N. sign(0) = 0 to avoid NaN.
        float sign_diff = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
        dL_dout_img[3 * pix_id + ch] = lambda_l1 * sign_diff * inv_num_pixels;
    }

    // Depth distortion regularizer — uniform per-pixel grad, sum the loss
    // value too (distortion is already accumulated per-pixel in the forward).
    float dist = out_distortion[pix_id];
    // Per-pixel normalization: mean over H*W. inv_num_pixels is 1/(H*W*3) so
    // multiply by 3 to get the per-pixel mean (avoiding another scalar).
    float inv_HW = inv_num_pixels * 3.0f;
    local_loss += lambda_dist * dist * inv_HW;
    dL_dout_distortion[pix_id] = lambda_dist * inv_HW;

    // DSSIM loss value: lambda_dssim * (1 - mean(ssim_map)).
    // ssim_sum is the sum over all H*W*3 entries; inv_num_pixels = 1/(H*W*3),
    // so mean(ssim_map) = ssim_sum[0] * inv_num_pixels. The full (1 - mean)
    // term must only be added once — pixel (0,0) carries the whole scalar so
    // we don't multi-count across threads. The DSSIM gradient on dL_dout_img
    // is applied by the separate ssim_backward kernels (this kernel only
    // writes L1's contribution; SSIM kernels add to it atomically).
    if (pix.x == 0u && pix.y == 0u) {
        float ssim_mean = ssim_sum[0] * inv_num_pixels;
        local_loss += lambda_dssim * (1.0f - ssim_mean);
    }

    atomic_fetch_add_explicit(loss_sum, local_loss, memory_order_relaxed);
}

// ===== DSSIM (Differentiable SSIM) — Colab parity for 2DGS training =====
// L1 alone lets the optimizer find streaky surfels aligned with view rays —
// pixels match but geometry is wrong. SSIM penalizes structural mismatch
// over an 11×11 Gaussian window, forcing locally-correct neighborhoods.
//
// Reference: hbb1/2d-gaussian-splatting utils/loss_utils.py.
// loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - mean(SSIM)).
// Default lambda_dssim = 0.2. C1 = 0.01², C2 = 0.03².
//
// The 2D Gaussian filter is implemented as a separable 1D pair: horizontal
// blur then vertical blur. This matches PyTorch's F.conv2d with padding=5,
// which uses zero padding for out-of-bounds reads.
//
// Forward dispatch sequence (called from msplat_train_step_2dgs):
//   1. ssim_prep_products_kernel:  build (img1*img1, img2*img2, img1*img2)
//      packed into one 9-channel tmp.
//   2. ssim_blur_horiz_kernel ×5:  horizontal blur of {img1, img2, img1²,
//      img2², img1*img2}.
//   3. ssim_blur_vert_kernel ×5:   vertical blur of the same five sources.
//   4. ssim_compute_kernel:        per-pixel-per-channel SSIM, atomic-sum.
//
// Backward dispatch sequence:
//   5. ssim_backward_compute_kernel: per-pixel gradients of SSIM w.r.t. its
//      five blurred inputs (writes grad_mu1, grad_mu2, grad_sigma1_sq,
//      grad_sigma2_sq, grad_sigma12; also folds the sigma-path corrections
//      back into grad_mu1 and grad_mu2).
//   6. ssim_blur_vert_kernel ×3 + ssim_blur_horiz_kernel ×3: re-blur the
//      grad_{mu1, sigma1_sq, sigma12} buffers (Gaussian is symmetric, so the
//      adjoint convolution is the same kernel).
//   7. ssim_backward_accumulate_kernel: combines the three blurred-grad
//      paths plus the (img1, img2) factors and atomically adds to
//      dL_dout_img (which already holds the L1 contribution).

constant int SSIM_WINDOW_SIZE = 11;
constant int SSIM_WINDOW_RADIUS = 5;  // (WINDOW_SIZE - 1) / 2

// 11-element 1D Gaussian kernel, sigma = 1.5, normalized so sum = 1.
// Matches PyTorch reference: `gauss = exp(-(x-5)² / (2·1.5²))`, then
// `gauss /= gauss.sum()` (utils/loss_utils.py). Unnormalized sum ≈ 3.7592328;
// normalized weights below sum to 1.0 within float32 round-off.
constant float SSIM_GAUSS_KERNEL[11] = {
    0.001028380084f,
    0.007598758135f,
    0.036000772128f,
    0.109360689510f,
    0.213005537711f,
    0.266011724862f,
    0.213005537711f,
    0.109360689510f,
    0.036000772128f,
    0.007598758135f,
    0.001028380084f
};

// Pack [img1, img2, img1*img1, img2*img2, img1*img2] per-pixel-per-channel into
// a single 15-channel tmp buffer? Cleaner: keep them as 5 separate [H,W,3]
// buffers — only img1*img1, img2*img2, img1*img2 are computed; img1 and img2
// are read directly from out_img and gt. This kernel writes only the product
// tensors.
kernel void ssim_prep_products_kernel(
    constant uint2& img_size,
    constant float* img1,        // out_img, [H, W, 3]
    constant float* img2,        // gt,      [H, W, 3]
    device   float* img1_sq,     // [H, W, 3]
    device   float* img2_sq,     // [H, W, 3]
    device   float* img1_img2,   // [H, W, 3]
    uint2 pix [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    if (pix.x >= W || pix.y >= H) return;
    uint base = (pix.y * W + pix.x) * 3u;
    for (uint c = 0u; c < 3u; ++c) {
        float a = img1[base + c];
        float b = img2[base + c];
        img1_sq  [base + c] = a * a;
        img2_sq  [base + c] = b * b;
        img1_img2[base + c] = a * b;
    }
}

// Separable Gaussian — horizontal pass. Output is the same shape as input.
// Zero-padding outside [0, W) to match PyTorch F.conv2d(padding=5).
kernel void ssim_blur_horiz_kernel(
    constant uint2& img_size,
    constant float* in_img,      // [H, W, 3]
    device   float* out_img,     // [H, W, 3]
    uint2 pix [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    if (pix.x >= W || pix.y >= H) return;
    int x = (int)pix.x;
    int y = (int)pix.y;

    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (int k = -SSIM_WINDOW_RADIUS; k <= SSIM_WINDOW_RADIUS; ++k) {
        int xs = x + k;
        if (xs < 0 || xs >= (int)W) continue;   // zero-pad outside
        float w = SSIM_GAUSS_KERNEL[k + SSIM_WINDOW_RADIUS];
        uint base = ((uint)y * W + (uint)xs) * 3u;
        acc[0] += w * in_img[base + 0];
        acc[1] += w * in_img[base + 1];
        acc[2] += w * in_img[base + 2];
    }
    uint out_base = ((uint)y * W + (uint)x) * 3u;
    out_img[out_base + 0] = acc[0];
    out_img[out_base + 1] = acc[1];
    out_img[out_base + 2] = acc[2];
}

// Separable Gaussian — vertical pass. Output is the same shape as input.
// Zero-padding outside [0, H) to match PyTorch F.conv2d(padding=5).
kernel void ssim_blur_vert_kernel(
    constant uint2& img_size,
    constant float* in_img,      // [H, W, 3]
    device   float* out_img,     // [H, W, 3]
    uint2 pix [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    if (pix.x >= W || pix.y >= H) return;
    int x = (int)pix.x;
    int y = (int)pix.y;

    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (int k = -SSIM_WINDOW_RADIUS; k <= SSIM_WINDOW_RADIUS; ++k) {
        int ys = y + k;
        if (ys < 0 || ys >= (int)H) continue;   // zero-pad outside
        float w = SSIM_GAUSS_KERNEL[k + SSIM_WINDOW_RADIUS];
        uint base = ((uint)ys * W + (uint)x) * 3u;
        acc[0] += w * in_img[base + 0];
        acc[1] += w * in_img[base + 1];
        acc[2] += w * in_img[base + 2];
    }
    uint out_base = ((uint)y * W + (uint)x) * 3u;
    out_img[out_base + 0] = acc[0];
    out_img[out_base + 1] = acc[1];
    out_img[out_base + 2] = acc[2];
}

// Per-pixel-per-channel SSIM map computation. Reads the five blurred buffers
// (mu1, mu2, blur(img1²), blur(img2²), blur(img1*img2)), computes
// sigma1_sq / sigma2_sq / sigma12 inline, and emits ssim_map[H,W,3].
// Also atomic-sums all (H*W*3) entries into ssim_sum[0] so the loss kernel
// can read a single scalar.
kernel void ssim_compute_kernel(
    constant uint2& img_size,
    constant float* mu1,             // blur(img1)        — [H, W, 3]
    constant float* mu2,             // blur(img2)        — [H, W, 3]
    constant float* blur_img1_sq,    // blur(img1*img1)   — [H, W, 3]
    constant float* blur_img2_sq,    // blur(img2*img2)   — [H, W, 3]
    constant float* blur_img1_img2,  // blur(img1*img2)   — [H, W, 3]
    device   float* ssim_map,        // [H, W, 3] — cached for backward
    device atomic<float>* ssim_sum,  // [1] scalar — sum over all entries
    uint2 pix [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    if (pix.x >= W || pix.y >= H) return;
    uint base = (pix.y * W + pix.x) * 3u;

    const float C1 = 0.0001f;   // (0.01)²
    const float C2 = 0.0009f;   // (0.03)²

    float local_sum = 0.0f;
    for (uint c = 0u; c < 3u; ++c) {
        float m1 = mu1[base + c];
        float m2 = mu2[base + c];
        float m1_sq = m1 * m1;
        float m2_sq = m2 * m2;
        float m1m2  = m1 * m2;

        // sigma_sq = blur(img²) - mu²;  sigma12 = blur(img1*img2) - mu1*mu2.
        float s1_sq = blur_img1_sq  [base + c] - m1_sq;
        float s2_sq = blur_img2_sq  [base + c] - m2_sq;
        float s12   = blur_img1_img2[base + c] - m1m2;

        float A = 2.0f * m1m2 + C1;
        float B = 2.0f * s12  + C2;
        float Cd = m1_sq + m2_sq + C1;
        float Dd = s1_sq + s2_sq + C2;

        float val = (A * B) / (Cd * Dd);
        ssim_map[base + c] = val;
        local_sum += val;
    }
    atomic_fetch_add_explicit(ssim_sum, local_sum, memory_order_relaxed);
}

// Per-pixel backward of SSIM, computed analytically. Reads the cached forward
// blurred buffers and writes the five blurred-input gradients. dL/dssim_map
// is uniform across all pixels and channels: -lambda_dssim * inv_num_pixels.
//
// We bake dL/dssim_map into each output gradient so the downstream re-blur
// passes don't need to multiply.
//
// The grad-mu paths each get TWO contributions: the direct A/C derivative,
// PLUS the sigma-path correction (sigma_sq = ... - mu²; sigma12 = ... - mu1*mu2).
// Both contributions are folded here so the downstream blur kernels see a
// single grad_mu1 / grad_mu2 buffer each.
kernel void ssim_backward_compute_kernel(
    constant uint2& img_size,
    constant float& dL_dssim_uniform,  // = -lambda_dssim * inv_num_pixels (scalar)
    constant float* mu1,
    constant float* mu2,
    constant float* blur_img1_sq,
    constant float* blur_img2_sq,
    constant float* blur_img1_img2,
    device   float* grad_mu1,          // [H, W, 3]
    device   float* grad_mu2,          // [H, W, 3]
    device   float* grad_sigma1_sq,    // [H, W, 3]
    device   float* grad_sigma2_sq,    // [H, W, 3]
    device   float* grad_sigma12,      // [H, W, 3]
    uint2 pix [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    if (pix.x >= W || pix.y >= H) return;
    uint base = (pix.y * W + pix.x) * 3u;

    const float C1 = 0.0001f;
    const float C2 = 0.0009f;
    float g = dL_dssim_uniform;   // == dL/d(ssim_map[i,j,c])

    for (uint c = 0u; c < 3u; ++c) {
        float m1 = mu1[base + c];
        float m2 = mu2[base + c];
        float m1_sq = m1 * m1;
        float m2_sq = m2 * m2;
        float m1m2  = m1 * m2;

        float s1_sq = blur_img1_sq  [base + c] - m1_sq;
        float s2_sq = blur_img2_sq  [base + c] - m2_sq;
        float s12   = blur_img1_img2[base + c] - m1m2;

        float A = 2.0f * m1m2 + C1;
        float B = 2.0f * s12  + C2;
        float Cd = m1_sq + m2_sq + C1;
        float Dd = s1_sq + s2_sq + C2;

        float CD = Cd * Dd;
        float AB = A * B;

        // SSIM = AB / CD.
        // d(SSIM)/dmu1 = (2*m2*B*CD - AB*2*m1*Dd) / (CD)²
        //              = 2/(CD) * (m2*B - m1*A*B/Cd)
        // d(SSIM)/dmu2 = 2/(CD) * (m1*B - m2*A*B/Cd)
        // d(SSIM)/d(sigma12)   = 2A / (CD)
        // d(SSIM)/d(sigma1_sq) = -AB / (Cd*Dd²) = -AB / (CD*Dd)
        // d(SSIM)/d(sigma2_sq) = -AB / (CD*Dd)
        float inv_CD  = 1.0f / CD;
        float inv_Cd  = 1.0f / Cd;
        float inv_Dd  = 1.0f / Dd;

        float dssim_dmu1 = 2.0f * inv_CD * (m2 * B - m1 * AB * inv_Cd);
        float dssim_dmu2 = 2.0f * inv_CD * (m1 * B - m2 * AB * inv_Cd);
        float dssim_ds12   =  2.0f * A * inv_CD;
        float dssim_ds1_sq = -AB * inv_CD * inv_Dd;
        float dssim_ds2_sq = -AB * inv_CD * inv_Dd;

        // Multiply each by upstream dL/dssim_map = g (uniform).
        float gmu1   = g * dssim_dmu1;
        float gmu2   = g * dssim_dmu2;
        float gs12   = g * dssim_ds12;
        float gs1_sq = g * dssim_ds1_sq;
        float gs2_sq = g * dssim_ds2_sq;

        // Sigma-path correction: sigma1_sq depends on mu1 as -mu1²
        // (chain rule: d(sigma1_sq)/dmu1 = -2*m1), and sigma12 depends on
        // mu1 as -mu1*mu2. Fold these into grad_mu1/grad_mu2 so the blur
        // backward only needs to chain over grad_mu* once.
        gmu1 += gs1_sq * (-2.0f * m1);
        gmu1 += gs12   * (-m2);
        gmu2 += gs2_sq * (-2.0f * m2);
        gmu2 += gs12   * (-m1);

        grad_mu1      [base + c] = gmu1;
        grad_mu2      [base + c] = gmu2;
        grad_sigma1_sq[base + c] = gs1_sq;
        grad_sigma2_sq[base + c] = gs2_sq;
        grad_sigma12  [base + c] = gs12;
    }
}

// After the three relevant grad_{mu1, sigma1_sq, sigma12} buffers have been
// re-blurred (separable transpose conv = same kernel since Gaussian is
// symmetric), accumulate them into dL_dout_img with the per-pixel chain-rule
// factors:
//   dL/dimg1 += blurred_grad_mu1                  (path: img1 → blur → mu1)
//   dL/dimg1 += 2 * img1 * blurred_grad_sigma1_sq (path: img1 → img1² → blur)
//   dL/dimg1 +=     img2 * blurred_grad_sigma12   (path: img1 → img1*img2 → blur)
// Existing dL_dout_img already contains the L1 gradient — we ADD here, not
// overwrite. We don't bother with atomic-add because each pixel is written
// by exactly one thread (no overlap).
kernel void ssim_backward_accumulate_kernel(
    constant uint2& img_size,
    constant float* img1,                  // out_img  — [H, W, 3]
    constant float* img2,                  // gt       — [H, W, 3]
    constant float* blurred_grad_mu1,      // [H, W, 3]
    constant float* blurred_grad_sigma1_sq,// [H, W, 3]
    constant float* blurred_grad_sigma12,  // [H, W, 3]
    device   float* dL_dout_img,           // [H, W, 3] — accumulated
    uint2 pix [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    if (pix.x >= W || pix.y >= H) return;
    uint base = (pix.y * W + pix.x) * 3u;

    for (uint c = 0u; c < 3u; ++c) {
        float a = img1[base + c];
        float b = img2[base + c];
        float gm1 = blurred_grad_mu1      [base + c];
        float gs1 = blurred_grad_sigma1_sq[base + c];
        float g12 = blurred_grad_sigma12  [base + c];

        // Sum the three paths' contributions to dL/dimg1.
        float contrib = gm1 + 2.0f * a * gs1 + b * g12;
        dL_dout_img[base + c] += contrib;
    }
}
// Single-dispatch inclusive prefix sum (cumsum) for int32 arrays.
// Uses one threadgroup: each thread serially sums its chunk, thread 0 scans
// block totals, then all threads write inclusive prefix sums.
// Used for small N (≤ PS_TG_SIZE) only; large N uses multi-threadgroup path.

#define PS_TG_SIZE 1024

kernel void prefix_sum_kernel(
    constant uint& N,
    constant int* input,
    device int* output,
    uint tg_tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    // Phase 1: Each thread serially sums its chunk
    uint chunk = (N + tg_size - 1) / tg_size;
    uint start = tg_tid * chunk;
    uint end = min(start + chunk, N);

    int my_sum = 0;
    for (uint i = start; i < end; i++) {
        my_sum += input[i];
    }

    // Phase 2: Two-level parallel prefix sum using SIMD
    // Level 1: intra-simdgroup exclusive prefix sum (hardware-accelerated)
    int sg_prefix = simd_prefix_exclusive_sum(my_sum);
    int sg_total = simd_sum(my_sum);

    // Level 2: cross-simdgroup scan (max 32 simdgroups for 1024 threads)
    uint num_sg = (tg_size + sg_size - 1) / sg_size;
    threadgroup int sg_totals[PS_TG_SIZE / 32];
    if (sg_lane == 0) {
        sg_totals[sg_id] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 scans simdgroup totals (max 32 iterations)
    threadgroup int sg_offsets[PS_TG_SIZE / 32];
    if (tg_tid == 0) {
        sg_offsets[0] = 0;
        for (uint i = 1; i < num_sg; i++) {
            sg_offsets[i] = sg_offsets[i - 1] + sg_totals[i - 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final prefix = cross-simdgroup offset + intra-simdgroup prefix
    int my_prefix = sg_offsets[sg_id] + sg_prefix;

    // Phase 3: Each thread writes inclusive prefix sum for its chunk
    int running = my_prefix;
    for (uint i = start; i < end; i++) {
        running += input[i];
        output[i] = running;
    }
}

// Multi-threadgroup prefix sum, pass 1: each threadgroup reduces its block of
// 1024 elements to a single total. Coalesced reads, 1 write per threadgroup.
kernel void block_reduce_kernel(
    constant uint& N,
    constant int* input,
    device int* block_totals,
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]]
) {
    uint idx = tg_id * PS_TG_SIZE + tg_tid;
    int val = (idx < N) ? input[idx] : 0;

    // Two-level reduction: SIMD sum → cross-SIMD sum
    int sg_total = simd_sum(val);

    threadgroup int sg_sums[PS_TG_SIZE / 32];
    if (sg_lane == 0) sg_sums[sg_id] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_tid == 0) {
        int total = 0;
        for (uint i = 0; i < PS_TG_SIZE / 32; i++) total += sg_sums[i];
        block_totals[tg_id] = total;
    }
}

// Multi-threadgroup prefix sum, pass 2: each threadgroup computes its block
// offset from block_totals, then writes inclusive prefix sums with coalesced access.
kernel void block_scan_propagate_kernel(
    constant uint& N,
    constant int* input,
    device int* output,
    constant int* block_totals,
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    // Step 1: Compute block offset (sum of all preceding block totals)
    int block_offset = 0;
    for (uint i = 0; i < tg_id; i++) {
        block_offset += block_totals[i];
    }

    // Step 2: Load element (coalesced)
    uint idx = tg_id * PS_TG_SIZE + tg_tid;
    int val = (idx < N) ? input[idx] : 0;

    // Step 3: Intra-block inclusive prefix sum (SIMD + cross-SIMD)
    int sg_prefix = simd_prefix_exclusive_sum(val);
    int sg_total = simd_sum(val);

    uint num_sg = PS_TG_SIZE / 32;
    threadgroup int sg_totals[PS_TG_SIZE / 32];
    threadgroup int sg_offsets[PS_TG_SIZE / 32];
    if (sg_lane == 0) sg_totals[sg_id] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_tid == 0) {
        int acc = 0;
        for (uint i = 0; i < num_sg; i++) {
            sg_offsets[i] = acc;
            acc += sg_totals[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Write inclusive prefix sum (coalesced)
    int inclusive = block_offset + sg_offsets[sg_id] + sg_prefix + val;
    if (idx < N) output[idx] = inclusive;
}

// ============================================================
// GPU Densification Kernels (kScaleDim-parametric, M2 reuses)
// ============================================================

#define DENSIFY_NOTHING 0
#define DENSIFY_SPLIT   1
#define DENSIFY_DUP     2

// Classify each gaussian as split, dup, or nothing based on gradient and scale thresholds.
kernel void densify_classify_kernel(
    constant int& N,
    constant float* xys_grad_norm    [[buffer(1)]],
    constant float* vis_counts       [[buffer(2)]],
    constant float* scales           [[buffer(3)]],  // [N, kScaleDim] log-space
    constant float* max_2d_size      [[buffer(4)]],
    constant float& half_max_dim     [[buffer(5)]],  // 0.5 * max(W,H)
    constant float& grad_thresh      [[buffer(6)]],
    constant float& size_thresh      [[buffer(7)]],
    constant float& screen_thresh    [[buffer(8)]],
    constant int& check_screen       [[buffer(9)]],
    device int* split_flag           [[buffer(10)]],
    device int* dup_flag             [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N) return;
    float vc = vis_counts[idx];
    if (vc <= 0.0f) { split_flag[idx] = 0; dup_flag[idx] = 0; return; }

    float avg_grad = (xys_grad_norm[idx] / vc) * half_max_dim;
    bool high_grad = avg_grad > grad_thresh;

    float max_scale = exp(scales[idx*kScaleDim]);
    for (int k = 1; k < kScaleDim; k++)
        max_scale = max(max_scale, exp(scales[idx*kScaleDim + k]));
    bool is_large = max_scale > size_thresh;

    bool do_split = is_large;
    if (check_screen && max_2d_size[idx] > screen_thresh) do_split = true;
    do_split = do_split && high_grad;

    bool do_dup = !is_large && high_grad;

    split_flag[idx] = do_split ? 1 : 0;
    dup_flag[idx]   = do_dup   ? 1 : 0;
}

// Append split children into backing buffers. One thread per original gaussian.
// Each split gaussian produces 2 children at [N + 2*(ord)], [N + 2*(ord)+1].
// Also shrinks parent scale by 1/1.6 and zeros optimizer state for children.
kernel void densify_append_split_kernel(
    constant int& N,
    constant int* split_flag         [[buffer(1)]],
    constant int* split_prefix       [[buffer(2)]],  // inclusive prefix sum
    constant float* random_samples   [[buffer(3)]],  // [2*N, 3] randn
    constant float& log_size_fac     [[buffer(4)]],  // log(1.6)
    device float* means_buf          [[buffer(5)]],
    device float* scales_buf         [[buffer(6)]],
    device float* quats_buf          [[buffer(7)]],
    device float* featuresDc_buf     [[buffer(8)]],
    device float* featuresRest_buf   [[buffer(9)]],
    device float* opacities_buf      [[buffer(10)]],
    constant int& fr_stride          [[buffer(11)]],  // featuresRest stride (e.g. 45)
    device float* adam_ea0           [[buffer(12)]],  // adam_exp_avg_buf[0..5]
    device float* adam_ea1           [[buffer(13)]],
    device float* adam_ea2           [[buffer(14)]],
    device float* adam_ea3           [[buffer(15)]],
    device float* adam_ea4           [[buffer(16)]],
    device float* adam_ea5           [[buffer(17)]],
    device float* adam_es0           [[buffer(18)]],  // adam_exp_avg_sq_buf[0..5]
    device float* adam_es1           [[buffer(19)]],
    device float* adam_es2           [[buffer(20)]],
    device float* adam_es3           [[buffer(21)]],
    device float* adam_es4           [[buffer(22)]],
    device float* adam_es5           [[buffer(23)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N || split_flag[idx] == 0) return;

    int ord = split_prefix[idx] - 1;  // 0-based ordinal among splits
    int c0 = N + 2 * ord;             // child 0 position
    int c1 = c0 + 1;                  // child 1 position

    // Read parent quaternion and normalize
    float qw = quats_buf[idx*4], qx = quats_buf[idx*4+1];
    float qy = quats_buf[idx*4+2], qz = quats_buf[idx*4+3];
    float qlen = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen;

    // Parent scale (exp). For 2DGS the third axis doesn't exist — zero it so
    // the random-sample offset has no out-of-plane component. Not a principled
    // 2DGS split (that would rotate the in-plane offset by the tangent frame
    // via the quat), but densify isn't dispatched in Milestone 1's render path;
    // the principled 2DGS split lands in Milestone 2.
    float sx = exp(scales_buf[idx*kScaleDim]);
    float sy = (kScaleDim > 1) ? exp(scales_buf[idx*kScaleDim+1]) : 0.0f;
    float sz = (kScaleDim > 2) ? exp(scales_buf[idx*kScaleDim+2]) : 0.0f;

    // For each of 2 children
    for (int k = 0; k < 2; k++) {
        int child = (k == 0) ? c0 : c1;
        int rand_idx = ord * 2 + k;

        // Scale random sample by parent scale
        float r0 = random_samples[rand_idx*3]   * sx;
        float r1 = random_samples[rand_idx*3+1] * sy;
        float r2 = random_samples[rand_idx*3+2] * sz;

        // Rotate by parent quaternion: v' = R @ v
        float v0 = (1-2*(qy*qy+qz*qz))*r0 + 2*(qx*qy-qw*qz)*r1 + 2*(qx*qz+qw*qy)*r2;
        float v1 = 2*(qx*qy+qw*qz)*r0 + (1-2*(qx*qx+qz*qz))*r1 + 2*(qy*qz-qw*qx)*r2;
        float v2 = 2*(qx*qz-qw*qy)*r0 + 2*(qy*qz+qw*qx)*r1 + (1-2*(qx*qx+qy*qy))*r2;

        // Child position = parent + rotated offset
        means_buf[child*3]   = means_buf[idx*3]   + v0;
        means_buf[child*3+1] = means_buf[idx*3+1] + v1;
        means_buf[child*3+2] = means_buf[idx*3+2] + v2;

        // Child scale = shrunk parent scale (kScaleDim entries — 3 for 3DGS, 2 for 2DGS)
        for (int j = 0; j < kScaleDim; j++)
            scales_buf[child*kScaleDim + j] = scales_buf[idx*kScaleDim + j] - log_size_fac;

        // Copy parent quaternion, featuresDc, opacities
        for (int j = 0; j < 4; j++) quats_buf[child*4+j] = quats_buf[idx*4+j];
        for (int j = 0; j < 3; j++) featuresDc_buf[child*3+j] = featuresDc_buf[idx*3+j];
        for (int j = 0; j < fr_stride; j++) featuresRest_buf[child*fr_stride+j] = featuresRest_buf[idx*fr_stride+j];
        opacities_buf[child] = opacities_buf[idx];

        // Zero optimizer state for children (strides: 3, kScaleDim, 4, 3, fr_stride, 1)
        for (int j = 0; j < 3; j++)          { adam_ea0[child*3+j] = 0;          adam_es0[child*3+j] = 0; }
        for (int j = 0; j < kScaleDim; j++)  { adam_ea1[child*kScaleDim+j] = 0;  adam_es1[child*kScaleDim+j] = 0; }
        for (int j = 0; j < 4; j++)          { adam_ea2[child*4+j] = 0;          adam_es2[child*4+j] = 0; }
        for (int j = 0; j < 3; j++)          { adam_ea3[child*3+j] = 0;          adam_es3[child*3+j] = 0; }
        for (int j = 0; j < fr_stride; j++)  { adam_ea4[child*fr_stride+j] = 0;  adam_es4[child*fr_stride+j] = 0; }
        adam_ea5[child] = 0; adam_es5[child] = 0;
    }

    // Shrink parent scale in-place
    for (int j = 0; j < kScaleDim; j++)
        scales_buf[idx*kScaleDim + j] -= log_size_fac;
}

// Append duplicate copies into backing buffers. One thread per original gaussian.
// Each dup produces 1 copy at [N + 2*nSplits + dup_ord].
kernel void densify_append_dup_kernel(
    constant int& N,
    constant int* dup_flag           [[buffer(1)]],
    constant int* dup_prefix         [[buffer(2)]],  // inclusive prefix sum
    constant int* split_prefix       [[buffer(3)]],  // to read nSplits = split_prefix[N-1]
    device float* means_buf          [[buffer(4)]],
    device float* scales_buf         [[buffer(5)]],
    device float* quats_buf          [[buffer(6)]],
    device float* featuresDc_buf     [[buffer(7)]],
    device float* featuresRest_buf   [[buffer(8)]],
    device float* opacities_buf      [[buffer(9)]],
    constant int& fr_stride          [[buffer(10)]],
    device float* adam_ea0           [[buffer(11)]],
    device float* adam_ea1           [[buffer(12)]],
    device float* adam_ea2           [[buffer(13)]],
    device float* adam_ea3           [[buffer(14)]],
    device float* adam_ea4           [[buffer(15)]],
    device float* adam_ea5           [[buffer(16)]],
    device float* adam_es0           [[buffer(17)]],
    device float* adam_es1           [[buffer(18)]],
    device float* adam_es2           [[buffer(19)]],
    device float* adam_es3           [[buffer(20)]],
    device float* adam_es4           [[buffer(21)]],
    device float* adam_es5           [[buffer(22)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N || dup_flag[idx] == 0) return;

    int nSplits = (N > 0) ? split_prefix[N - 1] : 0;
    int ord = dup_prefix[idx] - 1;
    int dst = N + 2 * nSplits + ord;

    // Copy all parent data
    for (int j = 0; j < 3; j++)         means_buf[dst*3+j]                 = means_buf[idx*3+j];
    for (int j = 0; j < kScaleDim; j++) scales_buf[dst*kScaleDim+j]        = scales_buf[idx*kScaleDim+j];
    for (int j = 0; j < 4; j++)         quats_buf[dst*4+j]                 = quats_buf[idx*4+j];
    for (int j = 0; j < 3; j++)         featuresDc_buf[dst*3+j]            = featuresDc_buf[idx*3+j];
    for (int j = 0; j < fr_stride; j++) featuresRest_buf[dst*fr_stride+j]  = featuresRest_buf[idx*fr_stride+j];
    opacities_buf[dst] = opacities_buf[idx];

    // Zero optimizer state (strides: 3, kScaleDim, 4, 3, fr_stride, 1)
    for (int j = 0; j < 3; j++)         { adam_ea0[dst*3+j] = 0;         adam_es0[dst*3+j] = 0; }
    for (int j = 0; j < kScaleDim; j++) { adam_ea1[dst*kScaleDim+j] = 0; adam_es1[dst*kScaleDim+j] = 0; }
    for (int j = 0; j < 4; j++)         { adam_ea2[dst*4+j] = 0;         adam_es2[dst*4+j] = 0; }
    for (int j = 0; j < 3; j++)         { adam_ea3[dst*3+j] = 0;         adam_es3[dst*3+j] = 0; }
    for (int j = 0; j < fr_stride; j++) { adam_ea4[dst*fr_stride+j] = 0; adam_es4[dst*fr_stride+j] = 0; }
    adam_ea5[dst] = 0; adam_es5[dst] = 0;
}

// Classify each post-growth gaussian as keep or cull.
// N_old = pre-growth count. N_new = N_old + 2*nSplits + nDups (computed from prefix sums).
// Dispatch with grid_size = worst_case (e.g. 3*N_old).
kernel void densify_cull_classify_kernel(
    constant int& N_old,
    constant int* split_prefix       [[buffer(1)]],  // [N_old] inclusive
    constant int* dup_prefix         [[buffer(2)]],  // [N_old] inclusive
    constant int* split_flag         [[buffer(3)]],  // [N_old] — marks split parents
    constant float* opacities_buf    [[buffer(4)]],
    constant float* scales_buf       [[buffer(5)]],
    constant float* max_2d_size      [[buffer(6)]],  // [N_old] only valid for idx < N_old
    constant float& cull_alpha_thresh [[buffer(7)]],  // 0.1
    constant float& cull_scale_thresh [[buffer(8)]],  // 0.5
    constant float& cull_screen_size  [[buffer(9)]],  // 0.15
    constant int& check_huge         [[buffer(10)]],
    constant int& check_screen       [[buffer(11)]],
    device int* keep_flag            [[buffer(12)]],
    uint idx [[thread_position_in_grid]]
) {
    int nSplits = (N_old > 0) ? split_prefix[N_old - 1] : 0;
    int nDups   = (N_old > 0) ? dup_prefix[N_old - 1] : 0;
    int N_new = N_old + 2 * nSplits + nDups;

    if (idx >= (uint)N_new) { keep_flag[idx] = 0; return; }

    // Sigmoid of opacity
    float opacity_sigmoid = 1.0f / (1.0f + exp(-opacities_buf[idx]));
    bool cull = opacity_sigmoid < cull_alpha_thresh;

    // Split parents are always culled
    if (idx < (uint)N_old && split_flag[idx] != 0) cull = true;

    // Huge gaussians
    if (check_huge) {
        float max_s = exp(scales_buf[idx*kScaleDim]);
        for (int k = 1; k < kScaleDim; k++)
            max_s = max(max_s, exp(scales_buf[idx*kScaleDim + k]));
        if (max_s > cull_scale_thresh) cull = true;
        if (check_screen && idx < (uint)N_old && max_2d_size[idx] > cull_screen_size) cull = true;
    }

    keep_flag[idx] = cull ? 0 : 1;
}

// Scatter kept elements from src to dst at compacted positions.
// One thread per float (elem * stride + sub).
kernel void compact_scatter_kernel(
    constant float* src              [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant int* keep_prefix        [[buffer(2)]],
    constant int* keep_flag          [[buffer(3)]],
    constant uint& N                 [[buffer(4)]],
    constant uint& stride            [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint elem = tid / stride;
    uint sub  = tid % stride;
    if (elem >= N || keep_flag[elem] == 0) return;
    int dst_elem = keep_prefix[elem] - 1;
    dst[dst_elem * stride + sub] = src[elem * stride + sub];
}

// Copy compacted data from scratch back to original buffer.
// Reads new_count from keep_prefix to determine bounds.
kernel void compact_copy_back_kernel(
    constant float* src              [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant int* keep_prefix        [[buffer(2)]],
    constant uint& last_prefix_idx   [[buffer(3)]],  // N_new - 1 (or worst_case - 1)
    constant uint& stride            [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int new_count = keep_prefix[last_prefix_idx];
    uint elem = tid / stride;
    uint sub  = tid % stride;
    if (elem >= (uint)new_count) return;
    dst[elem * stride + sub] = src[elem * stride + sub];
}

// ============================================================================
// Zero buffer kernel — replaces PyTorch .zero_() MPS dispatches.
// Writes 0 as uint32, which is the zero bit-pattern for float32, int32, etc.
// ============================================================================
kernel void zero_buffer_kernel(
    device uint* buf           [[buffer(0)]],
    constant uint& count       [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) buf[idx] = 0;
}
