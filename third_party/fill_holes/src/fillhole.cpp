#include <algorithm>
#include <map>
#include <torch/extension.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include "fillhole_kernel.h"
using namespace torch;
using namespace std;
vector<vector<int32_t>> build_boundary_loops1(const Tensor& boundary_edges) {
    auto edge_accessor = boundary_edges.accessor<int32_t, 2>();
    TORCH_CHECK(boundary_edges.device().is_cpu(), "Tensor must be on CPU");
    TORCH_CHECK(boundary_edges.dtype() == torch::kInt32, "Tensor must be int32");
    TORCH_CHECK(boundary_edges.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(boundary_edges.dim() == 2 && boundary_edges.size(1) == 2, "Tensor must be Nx2");
    std::map<int32_t, vector<int32_t>> edge_map;
    std::map<int32_t, int32_t> edge_stat;
    for (int64_t i = 0; i < boundary_edges.size(0); ++i) {
        int32_t v0 = edge_accessor[i][0];
        int32_t v1 = edge_accessor[i][1];
        edge_map[v0].push_back(v1);
        edge_stat[v0]=0;
        edge_stat[v1]=0;
    }
    unordered_set<int32_t> visited;
    vector<vector<int32_t>> loops;
    for (const auto& kv : edge_map) {
        int32_t start = kv.first;
        if (visited.count(start)) continue;

        vector<int32_t> loop;
        unordered_set<int32_t> temp_visited;
        int32_t curr = start;
        int32_t prev = -1;
        visited.insert(curr);
        loop.push_back(curr);
        temp_visited.insert(curr);
        int32_t next_v = -1;
        while (true) {
            const auto& neighbors = edge_map[curr];
            if (neighbors.empty()) break;
            next_v = -1;
            if (neighbors.size()==1 || prev==-1){
                next_v = neighbors[0];
            }
            else{
                int idx=edge_stat[curr]%2;
                next_v = neighbors[idx];

            }

            if (next_v == -1 || temp_visited.count(next_v)) break;
            loop.push_back(next_v);
            temp_visited.insert(next_v);
            prev = curr;
            curr = next_v;
        }
        if (next_v != loop[0]){
            edge_stat[next_v]=edge_stat[next_v]+1;
            continue;
        }
        if (loop.size() >= 3){
            for (int32_t v : loop) {
                edge_stat[v]=edge_stat[v]+1;
                visited.insert(v);
            }
            loops.push_back(loop);
        }

    }
    return loops;
}

Tensor fill_hole_cuda1(const Tensor& vertices, const Tensor& faces, const Tensor& boundary_edges,const Tensor& unique_edges) {
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must be (N, 3)");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be (M, 3)");
    TORCH_CHECK(boundary_edges.dim() == 2 && boundary_edges.size(1) == 2, "boundary_edges must be (K, 2)");
    TORCH_CHECK(faces.scalar_type() == torch::kInt32, "faces must be int32");
    TORCH_CHECK(boundary_edges.scalar_type() == torch::kInt32, "boundary_edges must be int32");
    vector<vector<int32_t>> loops = build_boundary_loops1(boundary_edges);
    if (loops.size() ==0){
        return torch::empty({0, 3}, torch::kInt32);
    }
    std::vector<int64_t> edges_keys(unique_edges.size(0));


    auto ptr = unique_edges.data_ptr<int32_t>();
    for (int i = 0; i < unique_edges.size(0); ++i) {
        int32_t u = ptr[i*2];
        int32_t v = ptr[i*2+1];
        int32_t a = std::min(u,v);
        int32_t b = std::max(u,v);
        edges_keys[i] = ((int64_t)a << 32) | b;
    }

    std::sort(edges_keys.begin(), edges_keys.end());
    auto filled_tensor = triangulate_loops_cuda(loops, vertices, edges_keys);
    return filled_tensor;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("fill_hole_cuda1", &fill_hole_cuda1, "Hole filling with torch tensor input cuda");
}
