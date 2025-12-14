/**
 * Minimal CUDA C++ Ray Tracer
 * - Single kernel that computes one pixel per thread
 * - Keeps data layout simple to reduce boilerplate
 *
 * Build (Windows + NVCC):
 *   nvcc -O3 ray_tracer_cuda.cu -o ray_tracer_cuda
 *
 * Run:
 *   ./ray_tracer_cuda              # complex scene
 *   ./ray_tracer_cuda --simple     # small scene
 *   ./ray_tracer_cuda --small      # 400x300 resolution
 *
 * Output:
 *   output_cuda.ppm
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

#define CUDA_CHECK(stmt)                                                      \
    do {                                                                      \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\\n",                 \
                          cudaGetErrorString(err), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
};
//inline lets compiler replace function call with direct code -> no call overhead.
__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ inline Vec3 operator*(float s, const Vec3& a) { return a * s; }
__host__ __device__ inline Vec3 operator/(const Vec3& a, float s) { return Vec3(a.x / s, a.y / s, a.z / s); }

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float length(const Vec3& a) { return sqrtf(dot(a, a)); }
__host__ __device__ inline Vec3 normalize(const Vec3& a) {
    float len = length(a);
    return (len > 0.0f) ? a / len : a;
}

struct Ray {
    Vec3 origin;
    Vec3 dir;
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    float reflectivity;
};

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity;
};


    // BOTTLENECK
//Ray-Sphere Intersection
//     Ray: P(t) = O + t·d
//     Sphere: |P - C|² = r²
__device__ bool intersect_sphere(const Ray& r, const Sphere* spheres, int idx, float t_min, float t_max, float& t_hit, Vec3& normal, Vec3& albedo) {
    const Sphere& s = spheres[idx];
    Vec3 oc = r.origin - s.center;
    float a = dot(r.dir, r.dir);
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b * b - a * c;
    if (disc < 0.0f) return false;
    float sq = sqrtf(disc);
    float t = (-b - sq) / a;
    if (t < t_min || t > t_max) {
        t = (-b + sq) / a;
        if (t < t_min || t > t_max) return false;
    }
    t_hit = t;
    Vec3 hit = r.origin + r.dir * t;
    normal = (hit - s.center) / s.radius;
    albedo = s.color;
    return true;
}

//Shading Function
__device__ Vec3 shade(const Vec3& point, const Vec3& normal, const Vec3& view_dir, const Sphere* spheres, int nspheres, const Light* lights, int nlights, const Vec3& ambient) {
    Vec3 color = ambient;
    for (int li = 0; li < nlights; ++li) {
        Vec3 light_dir = normalize(lights[li].position - point);
        float diff = fmaxf(0.0f, dot(normal, light_dir));
        color = color + lights[li].intensity * diff * Vec3(lights[li].color.x, lights[li].color.y, lights[li].color.z);
        Vec3 half_vec = normalize(light_dir - view_dir);
        float spec = powf(fmaxf(0.0f, dot(normal, half_vec)), 32.0f);
        color = color + spec * lights[li].intensity * Vec3(1.0f, 1.0f, 1.0f);
    }
    return color;
}

//Main Rendering Kernel
__global__ void render_kernel(Vec3* fb, int width, int height, const Sphere* spheres, int nspheres, const Light* lights, int nlights, Vec3 ambient) {
    // THREAD INDEXING
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / (width - 1);
    float v = (y + 0.5f) / (height - 1);
    // Camera (match OpenACC variant)
    Vec3 origin(13.0f, 2.0f, -15.0f);
    Vec3 target(0.0f, 0.0f, 0.0f);
    Vec3 forward = normalize(target - origin);
    Vec3 right = normalize(Vec3(forward.z, 0.0f, -forward.x));
    Vec3 up = normalize(Vec3(
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x));
    float fov = 40.0f * 3.14159265f / 180.0f;
    float half_h = tanf(fov / 2.0f);
    float aspect = (float)width / (float)height;
    float half_w = aspect * half_h;
    Vec3 pixel_dir = normalize(forward + (2.0f * u - 1.0f) * half_w * right + (2.0f * v - 1.0f) * half_h * up);
    Ray r{origin, pixel_dir};

    float closest = 1e20f;
    Vec3 n_hit, albedo;
    bool hit_any = false;
    for (int si = 0; si < nspheres; ++si) {
        float t_hit;
        Vec3 n_tmp, alb;
        if (intersect_sphere(r, spheres, si, 0.001f, closest, t_hit, n_tmp, alb)) {
            hit_any = true;
            closest = t_hit;
            n_hit = n_tmp;
            albedo = alb;
        }
    }

    Vec3 col;
    if (hit_any) {
        Vec3 point = r.origin + r.dir * closest;
        Vec3 view_dir = normalize(r.dir);
        col = shade(point, n_hit, view_dir, spheres, nspheres, lights, nlights, ambient);
        col = Vec3(col.x * albedo.x, col.y * albedo.y, col.z * albedo.z);
    } else {
        float t = 0.5f * (r.dir.y + 1.0f);
        col = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
    }

    col = Vec3(sqrtf(col.x), sqrtf(col.y), sqrtf(col.z));
    // Coalesced Memory Access -> Threads with consecutive x values write to consecutive memory addresses.
    fb[y * width + x] = col;
}

// Host utilities
static void write_ppm(const std::string& filename, const std::vector<Vec3>& fb, int w, int h) {
    FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "P3\n%d %d\n255\n", w, h);
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            const Vec3& c = fb[j * w + i];
            int r = static_cast<int>(255.999f * std::clamp(c.x, 0.0f, 1.0f));
            int g = static_cast<int>(255.999f * std::clamp(c.y, 0.0f, 1.0f));
            int b = static_cast<int>(255.999f * std::clamp(c.z, 0.0f, 1.0f));
            std::fprintf(f, "%d %d %d\n", r, g, b);
        }
    }
    std::fclose(f);
}

struct Scene {
    Sphere* spheres;
    Light* lights;
    int num_spheres;
    int num_lights;
    Vec3 ambient;
};

static Scene build_scene(bool complex_scene) {
    Scene scene;
    scene.ambient = Vec3(0.1f, 0.1f, 0.1f);
    
    if (complex_scene) {
        scene.num_spheres = 5;
        scene.num_lights = 3;
        scene.spheres = new Sphere[5];
        scene.lights = new Light[3];
        
        scene.spheres[0] = {Vec3(0, 0, 0), 2.0f, Vec3(1.0f, 0.3f, 0.3f), 0.5f};
        scene.spheres[1] = {Vec3(-4, 0, 2), 1.5f, Vec3(0.3f, 1.0f, 0.3f), 0.3f};
        scene.spheres[2] = {Vec3(3, -1, 1), 1.0f, Vec3(0.3f, 0.3f, 1.0f), 0.4f};
        scene.spheres[3] = {Vec3(2, 0.5f, -2), 1.2f, Vec3(0.8f, 0.8f, 0.8f), 0.9f};
        scene.spheres[4] = {Vec3(0, -102, 0), 100.0f, Vec3(0.5f, 0.5f, 0.5f), 0.1f};
        
        scene.lights[0] = {Vec3(10, 10, -10), Vec3(1, 1, 1), 0.8f};
        scene.lights[1] = {Vec3(-10, 10, -10), Vec3(1, 0.9f, 0.8f), 0.6f};
        scene.lights[2] = {Vec3(0, 5, -15), Vec3(0.8f, 0.8f, 1), 0.4f};
    } else {
        scene.num_spheres = 4;
        scene.num_lights = 1;
        scene.spheres = new Sphere[4];
        scene.lights = new Light[1];
        
        scene.spheres[0] = {Vec3(0, 0, 0), 1.0f, Vec3(1.0f, 0.3f, 0.3f), 0.5f};
        scene.spheres[1] = {Vec3(-2, 0, 1), 1.0f, Vec3(0.3f, 1.0f, 0.3f), 0.3f};
        scene.spheres[2] = {Vec3(2, 0, 1), 1.0f, Vec3(0.3f, 0.3f, 1.0f), 0.4f};
        scene.spheres[3] = {Vec3(0, -101, 0), 100.0f, Vec3(0.5f, 0.5f, 0.5f), 0.1f};
        
        scene.lights[0] = {Vec3(5, 5, -5), Vec3(1, 1, 1), 1.0f};
    }
    return scene;
}

int main(int argc, char** argv) {
    bool complex_scene = true;
    int width = 800;
    int height = 600;
    if (argc > 1 && std::string(argv[1]) == "--simple") {
        complex_scene = false;
    } else if (argc > 1 && std::string(argv[1]) == "--small") {
        width = 400;
        height = 300;
    }

    std::printf("CUDA Ray Tracer\n");
    std::printf("Resolution: %dx%d (%d pixels)\n", width, height, width * height);
    std::printf("Scene: %s\n", complex_scene ? "Complex" : "Simple");

    Scene scene = build_scene(complex_scene);
    const int npix = width * height;
    std::printf("Spheres: %d, Lights: %d\n", scene.num_spheres, scene.num_lights);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device buffers
    Vec3* d_fb = nullptr;
    Sphere* d_spheres = nullptr;
    Light* d_lights = nullptr;
    
    cudaEventRecord(start);
    
    CUDA_CHECK(cudaMalloc(&d_fb, npix * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_spheres, scene.num_spheres * sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&d_lights, scene.num_lights * sizeof(Light)));
    CUDA_CHECK(cudaMemcpy(d_spheres, scene.spheres, scene.num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lights, scene.lights, scene.num_lights * sizeof(Light), cudaMemcpyHostToDevice));

    //CUDA block = 16 × 16 = 256 threads
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    std::printf("\nLaunching kernel with grid(%d, %d) and block(%d, %d)...\n", 
                grid.x, grid.y, block.x, block.y);
    
    render_kernel<<<grid, block>>>(d_fb, width, height, d_spheres, scene.num_spheres, d_lights, scene.num_lights, scene.ambient);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::printf("GPU Render Time: %.3f seconds\n", milliseconds / 1000.0f);
    std::printf("Throughput: %.2f Mpixels/sec\n", (npix / (milliseconds / 1000.0f)) / 1e6);

    std::vector<Vec3> framebuffer(npix);
    CUDA_CHECK(cudaMemcpy(framebuffer.data(), d_fb, npix * sizeof(Vec3), cudaMemcpyDeviceToHost));
    write_ppm("output_cuda.ppm", framebuffer, width, height);
    std::printf("Output saved to: output_cuda.ppm\n");

    cudaFree(d_fb);
    cudaFree(d_spheres);
    cudaFree(d_lights);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    delete[] scene.spheres;
    delete[] scene.lights;
    
    return 0;
}

