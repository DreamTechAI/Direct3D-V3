#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
using namespace std;
__device__ float compute_angle_score_device1(const float3& prev, const float3& curr, const float3& next, const float3& normal) {
    float3 v1 = { curr.x - prev.x, curr.y - prev.y, curr.z - prev.z };
    float3 v2 = { next.x - curr.x, next.y - curr.y, next.z - curr.z };

    float len1 = sqrtf(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z) + 1e-8f;
    float len2 = sqrtf(v2.x*v2.x + v2.y*v2.y + v2.z*v2.z) + 1e-8f;
    float dot = -(v1.x*v2.x + v1.y*v2.y + v1.z*v2.z) / (len1 * len2);
    dot = fmaxf(-1.0f, fminf(1.0f, dot));

    float theta = acosf(dot);

    // 计算叉积
    float3 cross = {
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    };

    // 判断凹凸角
    float orient = cross.x * normal.x + cross.y * normal.y + cross.z * normal.z;
    if (orient < 0.0f) {
        theta = 2.0f * M_PI - theta;
    }

    return theta;
}
__device__ float3 compute_oriented_loop_normal(
    const int32_t* loops_flat,
    int loop_start,
    int loop_len,
    const float3* vertices
) {
    float3 normal = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < loop_len; ++i) {
        int vi = loops_flat[loop_start + i];
        int vj = loops_flat[loop_start + (i + 1) % loop_len];

        const float3& v0 = vertices[vi];
        const float3& v1 = vertices[vj];

        // 累加 v0 × v1
        normal.x += v0.y * v1.z - v0.z * v1.y;
        normal.y += v0.z * v1.x - v0.x * v1.z;
        normal.z += v0.x * v1.y - v0.y * v1.x;
    }

    float len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z) + 1e-12f;
    normal.x /= len;
    normal.y /= len;
    normal.z /= len;
    return normal;
}

__device__ bool edge_exists(int32_t u, int32_t v, const int64_t* edges_keys, int num_edges) {
    // 将边标准化为 key
    int64_t key = ((int64_t)min(u,v) << 32) | max(u,v);

    int left = 0;
    int right = num_edges - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;  // 避免溢出
        uint64_t mid_key = edges_keys[mid];

        if (mid_key == key) return true;
        else if (mid_key < key) left = mid + 1;
        else right = mid - 1;
    }
    return false;
}
__global__ void ear_clipping_batch_kernel(
    int32_t* loops_flat,
    const int32_t* loop_offsets,
    const int32_t* loop_sizes,
    const float3* vertices,
    const int32_t total_loops,
    int32_t* tris_out,
    const int64_t* edges_keys_gpu,
    const int32_t num_edges
) {

    int loop_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (loop_id<total_loops){
    int loop_start = loop_offsets[loop_id];
    int loop_len   = loop_sizes[loop_id];
    int tris_offset = 0;
    for (int i = 0; i < loop_id; ++i) tris_offset += loop_sizes[i] - 2; // prefix sum (small loops ok)
    float3 loop_normal = compute_oriented_loop_normal(
            loops_flat, loop_start, loop_len, vertices
        );
    int remain = loop_len;
    int tris_count = 0;
    while (remain > 2) {
            // 计算所有角度并排序找最小角度顶点
            // 这里用简单的线性扫描实现
            float min_angle = 1e9f;
            int min_idx = -1;

            for (int i = 0; i < remain; ++i) {
                int i_prev = loops_flat[loop_start+(i - 1 + remain) % remain];
                int i_curr = loops_flat[loop_start+i];
                int i_next = loops_flat[loop_start+(i + 1) % remain];
                float angle = compute_angle_score_device1(vertices[i_prev], vertices[i_curr], vertices[i_next],loop_normal);
                if (edge_exists(i_prev, i_next, edges_keys_gpu, num_edges)) {
                    angle += 10.0f;
                }
                if (angle < min_angle) {
                    min_angle = angle;
                    min_idx = i;
                }

            }

            // 输出三角形
            int i_prev = loops_flat[loop_start+(min_idx - 1 + remain) % remain];
            int i_curr = loops_flat[loop_start+min_idx];
            int i_next = loops_flat[loop_start+(min_idx + 1) % remain];

            int out_idx = (tris_offset + tris_count) * 3;
            tris_out[out_idx + 0] = i_prev;
            tris_out[out_idx + 1] = i_curr;
            tris_out[out_idx + 2] = i_next;
            tris_count++;

            // 删除顶点 (shift左移覆盖)
            for (int i = min_idx; i < remain - 1; ++i) {
                loops_flat[loop_start+i] = loops_flat[loop_start+i + 1];
            }
            remain--;
        }
    }
}

torch::Tensor triangulate_loops_cuda(
    const vector<vector<int32_t>>& loops,
    const torch::Tensor& vertices_cpu,
    const vector<int64_t>& edges_keys
) {
    TORCH_CHECK(vertices_cpu.device().is_cpu(), "vertices must be on CPU");
    TORCH_CHECK(vertices_cpu.size(1) == 3, "vertices must be Nx3");

    // 展平 loops
    vector<int32_t> loops_flat;
    vector<int32_t> loop_offsets;
    vector<int32_t> loop_sizes;
    int offset = 0;
    int total_tris = 0;
    int total_loops = 0;
    for (auto& lp : loops) {
        loop_offsets.push_back(offset);
        loop_sizes.push_back(lp.size());
        loops_flat.insert(loops_flat.end(), lp.begin(), lp.end());
        offset += lp.size();
        total_tris += lp.size() - 2;
        total_loops+=1;
    }

    // 上传数据
    auto vertices_gpu = vertices_cpu.to(torch::kCUDA).contiguous();
    auto verts_ptr = reinterpret_cast<float3*>(vertices_gpu.data_ptr<float>());

    auto loops_flat_gpu = torch::from_blob(loops_flat.data(), {(int)loops_flat.size()}, torch::kInt32).clone().to(torch::kCUDA);
    auto loop_offsets_gpu = torch::from_blob(loop_offsets.data(), {(int)loop_offsets.size()}, torch::kInt32).clone().to(torch::kCUDA);
    auto loop_sizes_gpu = torch::from_blob(loop_sizes.data(), {(int)loop_sizes.size()}, torch::kInt32).clone().to(torch::kCUDA);

    auto tris_gpu = torch::empty({total_tris, 3},torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto edges_keys_gpu = torch::from_blob(const_cast<int64_t*>(edges_keys.data()),{static_cast<int64_t>(edges_keys.size())},torch::kInt64).clone().to(torch::kCUDA);

    int num_edges = edges_keys.size();
    int threads = 64;
    // 启动 kernel
    ear_clipping_batch_kernel<<<(total_loops+ threads - 1)/threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        loops_flat_gpu.data_ptr<int32_t>(),
        loop_offsets_gpu.data_ptr<int32_t>(),
        loop_sizes_gpu.data_ptr<int32_t>(),
        verts_ptr,
        total_loops,
        tris_gpu.data_ptr<int32_t>(),
        edges_keys_gpu.data_ptr<int64_t>(),
        num_edges
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return tris_gpu.cpu();
}
