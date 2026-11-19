#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

struct Point3D {
    float x, y, z;
};

struct Triangle {
    Point3D v0, v1, v2;
};
__device__ Point3D cross(const Point3D& v1, const Point3D& v2) {
    Point3D result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return result;
}

// Compute the dot product of two vectors
__device__ float dot(const Point3D& v1, const Point3D& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// Subtract two 3D points (vector subtraction)
__device__ Point3D subtract(const Point3D& p1, const Point3D& p2) {
    Point3D result = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
    return result;
}
__device__ bool overlapIntervals(float min1, float max1, float min2, float max2) {
    return !(min1 > max2 || max1 < min2);
}

// 计算向量绝对值
__device__ Point3D absVec(const Point3D& v) {
    return {fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

// 根据轴投影三角形顶点，返回区间[min_proj, max_proj]
__device__ void projectTriangle(const Point3D tri[3], const Point3D& axis, float& min_proj, float& max_proj) {
    float p0 = dot(tri[0], axis);
    float p1 = dot(tri[1], axis);
    float p2 = dot(tri[2], axis);
    min_proj = fminf(p0, fminf(p1, p2));
    max_proj = fmaxf(p0, fmaxf(p1, p2));
}

// 判断三角形和立方体在 axis 轴上是否投影重叠
__device__ bool overlapOnAxis(const Point3D tri[3], const Point3D& box_half, const Point3D& axis) {
    // 跳过零向量
    if (axis.x == 0 && axis.y == 0 && axis.z == 0) return true;

    float min_proj, max_proj;
    projectTriangle(tri, axis, min_proj, max_proj);

    Point3D a = absVec(axis);
    float r = box_half.x * a.x + box_half.y * a.y + box_half.z * a.z;

    return overlapIntervals(min_proj, max_proj, -r, r);
}
__device__ bool triangleIntersectsCube(const Triangle& tri, const Point3D& cube_min,const int DIM) {
    Point3D box_center = {cube_min.x + 0.5f/(DIM-1), cube_min.y + 0.5f/(DIM-1), cube_min.z + 0.5f/(DIM-1)};
    Point3D box_half = {0.5f/(DIM-1), 0.5f/(DIM-1), 0.5f/(DIM-1)};

    // 将三角形顶点移到立方体局部坐标系
    Point3D tv[3];
    tv[0] = subtract(tri.v0, box_center);
    tv[1] = subtract(tri.v1, box_center);
    tv[2] = subtract(tri.v2, box_center);

    // 三角形边向量
    Point3D e0 = subtract(tv[1], tv[0]);
    Point3D e1 = subtract(tv[2], tv[1]);
    Point3D e2 = subtract(tv[0], tv[2]);

    // 13个分离轴
    Point3D axes[13] = {
        // 立方体的3个轴
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        // 三角形法线
        cross(e0, e1),
        // 边和立方体轴的叉积（9个）
        {0, -e0.z, e0.y}, {0, -e1.z, e1.y}, {0, -e2.z, e2.y},
        {e0.z, 0, -e0.x}, {e1.z, 0, -e1.x}, {e2.z, 0, -e2.x},
        {-e0.y, e0.x, 0}, {-e1.y, e1.x, 0}, {-e2.y, e2.x, 0}
    };

    // 测试13个轴是否分离
    for (int i = 0; i < 13; ++i) {
        if (!overlapOnAxis(tv, box_half, axes[i])) {
            return false; // 存在分离轴，三角形与立方体不相交
        }
    }

    return true; // 所有轴都有重叠，说明相交
}

__device__ void updateSDF(Triangle t, bool* voxel, const int DIM, const float threshold) {
    // Compute the bounding box of the triangle
    float minX = fminf(fminf(t.v0.x, t.v1.x), t.v2.x);
    float minY = fminf(fminf(t.v0.y, t.v1.y), t.v2.y);
    float minZ = fminf(fminf(t.v0.z, t.v1.z), t.v2.z);
    float maxX = fmaxf(fmaxf(t.v0.x, t.v1.x), t.v2.x);
    float maxY = fmaxf(fmaxf(t.v0.y, t.v1.y), t.v2.y);
    float maxZ = fmaxf(fmaxf(t.v0.z, t.v1.z), t.v2.z);

    // Convert bounding box to grid coordinates
    int iMin = max(0, (int)floorf((minX + 0.5)  * (DIM-1)));
    int jMin = max(0, (int)floorf((minY + 0.5)  * (DIM-1)));
    int kMin = max(0, (int)floorf((minZ + 0.5)  * (DIM-1)));
    int iMax = min(DIM - 1, (int)floorf((maxX + 0.5)  * (DIM-1)));
    int jMax = min(DIM - 1, (int)floorf((maxY + 0.5)  * (DIM-1)));
    int kMax = min(DIM - 1, (int)floorf((maxZ + 0.5)  * (DIM-1)));

    int range = (int)(threshold + 1);

    // Make the bounding box larger than the original
    iMax = min(DIM - 1, iMax + range);
    iMin = max(0, iMin - range);
    jMax = min(DIM - 1, jMax + range);
    jMin = max(0, jMin - range);
    kMax = min(DIM - 1, kMax + range);
    kMin = max(0, kMin - range);

    // Update the valid grids within the bounding box
    for (int i = iMin; i <= iMax; ++i) {
        for (int j = jMin; j <= jMax; ++j) {
            for (int k = kMin; k <= kMax; ++k) {
                Point3D queryPoint = {(float)i/(DIM-1) - 0.5, (float)j/(DIM-1) - 0.5, (float)k/(DIM-1) -0.5};
                bool insect=triangleIntersectsCube(t,queryPoint,DIM);
                if (insect){
                    uint32_t idx = static_cast<uint32_t>(i) * DIM * DIM + j * DIM + k;
                    voxel[idx]=insect;
                }
	    }
        }
    }
}

__global__ void compute_sdf_kernel(float* vertices, int* faces, bool * voxel, int numTriangles, const int DIM, const float threshold) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < numTriangles) {
        int f0 = faces[t * 3 + 0];
        int f1 = faces[t * 3 + 1];
        int f2 = faces[t * 3 + 2];
        Point3D v0 = {vertices[f0 * 3 + 0], vertices[f0 * 3 + 1], vertices[f0 * 3 + 2]};
        Point3D v1 = {vertices[f1 * 3 + 0], vertices[f1 * 3 + 1], vertices[f1 * 3 + 2]};
        Point3D v2 = {vertices[f2 * 3 + 0], vertices[f2 * 3 + 1], vertices[f2 * 3 + 2]};
        Triangle triangle = {v0, v1, v2};
        updateSDF(triangle, voxel, DIM, threshold);
    }
}

void compute_valid_sdf_cuda(float* vertices, int* faces, bool* voxel, int numTriangles, const int DIM=512, const float threshold=8) {
    int blockSize = 256;
    int gridSize = (numTriangles + blockSize - 1) / blockSize;

    // Launch the kernel
    compute_sdf_kernel<<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(vertices, faces, voxel, numTriangles, DIM, threshold);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
