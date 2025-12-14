/**
 * Minimal OpenACC Ray Tracer
 * - Focuses on the pixel loop + ray casting parallelization
 * - Keeps data layout GPU-friendly (plain structs/arrays)
 *
 * Build (NVHPC / nvc++ on Windows or WSL):
 *   nvc++ -acc -fast -Minfo=accel ray_tracer_openacc.cpp -o ray_tracer_openacc
 *
 * Run:
 *   ./ray_tracer_openacc              # complex scene (default)
 *   ./ray_tracer_openacc --simple     # small scene
 *   ./ray_tracer_openacc --small      # 400x300 resolution
 *
 * Output:
 *   output_acc.ppm
 */

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
inline Vec3 operator*(float s, const Vec3& a) { return a * s; }
inline Vec3 operator/(const Vec3& a, float s) { return Vec3(a.x / s, a.y / s, a.z / s); }

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) {
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
    float reflectivity;  // kept for parity, but not used in this minimal variant
};

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity;
};

struct Scene {
    std::vector<Sphere> spheres;
    std::vector<Light> lights;
    Vec3 ambient = Vec3(0.1f, 0.1f, 0.1f);
};

// Plain PPM writer
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

// Small helper routines that can be inlined on device
#pragma acc routine seq
static bool intersect_sphere(const Ray& r, const Sphere* spheres, int idx, float t_min, float t_max, float& t_hit, Vec3& normal, Vec3& albedo) {
    const Sphere& s = spheres[idx];
    Vec3 oc = r.origin - s.center;
    float a = dot(r.dir, r.dir);
    float b = dot(oc, r.dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b * b - a * c;
    if (disc < 0.0f) return false;
    float sq = std::sqrt(disc);
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

#pragma acc routine seq
static Vec3 shade(const Vec3& point, const Vec3& normal, const Vec3& view_dir, const Sphere* spheres, int nspheres, const Light* lights, int nlights, const Vec3& ambient) {
    Vec3 color = Vec3(ambient.x, ambient.y, ambient.z);
    for (int li = 0; li < nlights; ++li) {
        Vec3 light_dir = normalize(lights[li].position - point);
        float diff = std::max(0.0f, dot(normal, light_dir));
        color = color + lights[li].intensity * diff * Vec3(
            lights[li].color.x,
            lights[li].color.y,
            lights[li].color.z);
        // Simple specular
        Vec3 half_vec = normalize(light_dir - view_dir);
        float spec = std::pow(std::max(0.0f, dot(normal, half_vec)), 32.0f);
        color = color + spec * lights[li].intensity * Vec3(1.0f, 1.0f, 1.0f);
    }
    return color;
}

static void render_acc(std::vector<Vec3>& fb, int width, int height, const Scene& scene) {
    const int npix = width * height;
    fb.assign(npix, Vec3());

    // Flatten data for OpenACC
    const int nspheres = static_cast<int>(scene.spheres.size());
    const int nlights = static_cast<int>(scene.lights.size());
    // Capture raw pointers for device copy
    const Sphere* spheres = scene.spheres.data();
    const Light* lights = scene.lights.data();

#pragma acc data copyout(fb[0:npix]) copyin(spheres[0:nspheres], lights[0:nlights])
    {
        // Parallelize pixel traversal
#pragma acc parallel loop collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float u = (x + 0.5f) / (width - 1);
                float v = (y + 0.5f) / (height - 1);
                // Camera setup (simple pinhole)
                Vec3 origin(13.0f, 2.0f, -15.0f);
                Vec3 target(0.0f, 0.0f, 0.0f);
                Vec3 forward = normalize(target - origin);
                Vec3 right = normalize(Vec3(forward.z, 0.0f, -forward.x));
                Vec3 up = normalize(Vec3(
                    right.y * forward.z - right.z * forward.y,
                    right.z * forward.x - right.x * forward.z,
                    right.x * forward.y - right.y * forward.x));
                float fov = 40.0f * 3.14159265f / 180.0f;
                float half_h = std::tan(fov / 2.0f);
                float aspect = static_cast<float>(width) / static_cast<float>(height);
                float half_w = aspect * half_h;
                Vec3 pixel_dir = normalize(forward + (2.0f * u - 1.0f) * half_w * right + (2.0f * v - 1.0f) * half_h * up);
                Ray r{origin, pixel_dir};

                float closest = 1e20f;
                Vec3 n_hit, albedo;
                bool hit_any = false;
                // Simple linear traversal of spheres
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
                    col = shade(point, n_hit, view_dir, spheres, nspheres, lights, nlights, scene.ambient);
                    col = Vec3(col.x * albedo.x, col.y * albedo.y, col.z * albedo.z);
                } else {
                    // Sky gradient
                    float t = 0.5f * (r.dir.y + 1.0f);
                    col = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
                }

                // Gamma correction
                col = Vec3(std::sqrt(col.x), std::sqrt(col.y), std::sqrt(col.z));
                fb[y * width + x] = col;
            }
        }
    }
}

static Scene build_scene(bool complex_scene) {
    Scene scene;
    if (complex_scene) {
        scene.spheres.push_back({Vec3(0, 0, 0), 2.0f, Vec3(1.0f, 0.3f, 0.3f), 0.5f});
        scene.spheres.push_back({Vec3(-4, 0, 2), 1.5f, Vec3(0.3f, 1.0f, 0.3f), 0.3f});
        scene.spheres.push_back({Vec3(3, -1, 1), 1.0f, Vec3(0.3f, 0.3f, 1.0f), 0.4f});
        scene.spheres.push_back({Vec3(2, 0.5f, -2), 1.2f, Vec3(0.8f, 0.8f, 0.8f), 0.9f});
        // Ground
        scene.spheres.push_back({Vec3(0, -102, 0), 100.0f, Vec3(0.5f, 0.5f, 0.5f), 0.1f});
        // Lights
        scene.lights.push_back({Vec3(10, 10, -10), Vec3(1, 1, 1), 0.8f});
        scene.lights.push_back({Vec3(-10, 10, -10), Vec3(1, 0.9f, 0.8f), 0.6f});
        scene.lights.push_back({Vec3(0, 5, -15), Vec3(0.8f, 0.8f, 1), 0.4f});
    } else {
        scene.spheres.push_back({Vec3(0, 0, 0), 1.0f, Vec3(1.0f, 0.3f, 0.3f), 0.5f});
        scene.spheres.push_back({Vec3(-2, 0, 1), 1.0f, Vec3(0.3f, 1.0f, 0.3f), 0.3f});
        scene.spheres.push_back({Vec3(2, 0, 1), 1.0f, Vec3(0.3f, 0.3f, 1.0f), 0.4f});
        scene.spheres.push_back({Vec3(0, -101, 0), 100.0f, Vec3(0.5f, 0.5f, 0.5f), 0.1f});
        scene.lights.push_back({Vec3(5, 5, -5), Vec3(1, 1, 1), 1.0f});
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

    Scene scene = build_scene(complex_scene);
    std::vector<Vec3> framebuffer;
    render_acc(framebuffer, width, height, scene);
    write_ppm("output_acc.ppm", framebuffer, width, height);
    return 0;
}

