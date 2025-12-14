/**
 * High-Performance Ray Tracer
 * File: ray_tracer.cpp
 * 
 * Compile with:
 * g++ -std=c++17 -Wall ray_tracer.cpp -o ray_tracer_profile -lm
 */

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <limits>
#include <memory>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>


// Performance counters
struct PerformanceCounters {
    std::atomic<size_t> total_rays{0};
    std::atomic<size_t> primary_rays{0};
    std::atomic<size_t> shadow_rays{0};
    std::atomic<size_t> reflection_rays{0};
    std::atomic<size_t> intersection_tests{0};
    std::atomic<size_t> intersection_hits{0};
    std::atomic<size_t> pixels_processed{0};
    
    void reset() {
        total_rays = 0;
        primary_rays = 0;
        shadow_rays = 0;
        reflection_rays = 0;
        intersection_tests = 0;
        intersection_hits = 0;
        pixels_processed = 0;
    }
    
    void print() const {
        std::cout << "\n=== Performance Statistics ===" << std::endl;
        std::cout << "Total rays cast: " << total_rays << std::endl;
        std::cout << "  Primary rays: " << primary_rays << std::endl;
        std::cout << "  Shadow rays: " << shadow_rays << std::endl;
        std::cout << "  Reflection rays: " << reflection_rays << std::endl;
        std::cout << "Intersection tests: " << intersection_tests << std::endl;
        std::cout << "Intersection hits: " << intersection_hits << std::endl;
        double hit_rate = (intersection_tests > 0) ? 
            (100.0 * intersection_hits / intersection_tests) : 0.0;
        std::cout << "Hit rate: " << hit_rate << "%" << std::endl;
        std::cout << "Pixels processed: " << pixels_processed << std::endl;
        double rpp = (pixels_processed > 0) ? 
            ((double)total_rays / pixels_processed) : 0.0;
        std::cout << "Rays per pixel: " << rpp << std::endl;
    }
};

PerformanceCounters g_counters;

/**
 * 3D Vector class
 */
class Vec3 {
public:
    double x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(double t) const { return Vec3(x * t, y * t, z * t); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator/(double t) const { return Vec3(x / t, y / t, z / t); }
    
    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    
    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    
    double length() const { return std::sqrt(x * x + y * y + z * z); }
    double length_squared() const { return x * x + y * y + z * z; }
    
    Vec3 normalize() const {
        double len = length();
        return (len > 0) ? Vec3(x / len, y / len, z / len) : *this;
    }
    
    Vec3 reflect(const Vec3& normal) const {
        return *this - normal * (2.0 * this->dot(normal));
    }
};

/**
 * Ray structure
 */
struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    Ray() {}
    Ray(const Vec3& origin, const Vec3& direction) 
        : origin(origin), direction(direction) {}
    
    Vec3 at(double t) const { return origin + direction * t; }
};

/**
 * Material properties
 */
struct Material {
    Vec3 albedo;
    double reflectivity;
    double roughness;
    double metallic;
    
    Material(const Vec3& color = Vec3(0.5, 0.5, 0.5), 
             double refl = 0.0, double rough = 0.5, double metal = 0.0)
        : albedo(color), reflectivity(refl), roughness(rough), metallic(metal) {}
};

/**
 * Sphere primitive - MOST EXPENSIVE OPERATION
 */
class Sphere {
public:
    Vec3 center;
    double radius;
    Material material;
    
    Sphere(const Vec3& c, double r, const Material& m)
        : center(c), radius(r), material(m) {}
    
    // BOTTLENECK
//     Ray: P(t) = O + t·d
//     Sphere: |P - C|² = r²
    inline bool intersect(const Ray& ray, double t_min, double t_max, double& t) const {
        g_counters.intersection_tests++;
        
        Vec3 oc = ray.origin - center;                    // 3 subtractions
        double a = ray.direction.length_squared();        // 3 multiplications, 2 additions
        double half_b = oc.dot(ray.direction);           // 3 multiplications, 2 additions
        double c = oc.length_squared() - radius * radius; // 3 multiplications, 3 additions, 1 subtraction

        
        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0) {
            return false;
        }
        
        double sqrtd = std::sqrt(discriminant);
        double root = (-half_b - sqrtd) / a;
        
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max) {
                return false;
            }
        }
        
        t = root;
        g_counters.intersection_hits++;
        return true;
    }
    
    Vec3 normal_at(const Vec3& point) const {
        return (point - center) / radius;
    }
};

/**
 * Light source
 */
struct Light {
    Vec3 position;
    Vec3 color;
    double intensity;
    
    Light(const Vec3& pos, const Vec3& col, double intens)
        : position(pos), color(col), intensity(intens) {}
};

/**
 * Scene container
 */
class Scene {
public:
    std::vector<std::unique_ptr<Sphere>> spheres;
    std::vector<Light> lights;
    Vec3 ambient_light;
    
    Scene() : ambient_light(0.1, 0.1, 0.1) {}
    
    void add_sphere(const Vec3& center, double radius, const Material& mat) {
        spheres.push_back(std::make_unique<Sphere>(center, radius, mat));
    }
    
    void add_light(const Vec3& position, const Vec3& color, double intensity) {
        lights.emplace_back(position, color, intensity);
    }
    
    // BOTTLENECK
    inline bool intersect(const Ray& ray, double t_min, double t_max, 
                   double& t, const Sphere*& hit_sphere) const {
        double closest_t = t_max;
        bool hit_anything = false;
        
        for (const auto& sphere : spheres) {
            double temp_t;
            if (sphere->intersect(ray, t_min, closest_t, temp_t)) {
                hit_anything = true;
                closest_t = temp_t;
                hit_sphere = sphere.get();
            }
        }
        
        t = closest_t;
        return hit_anything;
    }
    
    void create_complex_scene() {
        // Main spheres
        add_sphere(Vec3(0, 0, 0), 2.0, 
                  Material(Vec3(1.0, 0.3, 0.3), 0.5, 0.2, 0.0));
        add_sphere(Vec3(-4, 0, 2), 1.5, 
                  Material(Vec3(0.3, 1.0, 0.3), 0.3, 0.5, 0.0));
        add_sphere(Vec3(3, -1, 1), 1.0, 
                  Material(Vec3(0.3, 0.3, 1.0), 0.4, 0.3, 0.0));
        add_sphere(Vec3(2, 0.5, -2), 1.2, 
                  Material(Vec3(0.8, 0.8, 0.8), 0.9, 0.1, 1.0));
        
        // Grid of smaller spheres
        for (int i = -5; i <= 5; i++) {
            for (int j = -3; j <= 3; j++) {
                if (std::abs(i) < 2 && std::abs(j) < 1) continue;
                
                double x = i * 2.5;
                double z = j * 2.5 + 8;
                double y = std::sin(i * 0.5) * std::cos(j * 0.5) * 1.5;
                
                Vec3 color(
                    0.5 + 0.5 * std::sin(i),
                    0.5 + 0.5 * std::cos(j),
                    0.5 + 0.5 * std::sin(i + j)
                );
                
                add_sphere(Vec3(x, y, z), 0.4, 
                          Material(color, 0.2, 0.6, 0.0));
            }
        }
        
        // Ground plane
        add_sphere(Vec3(0, -102, 0), 100, 
                  Material(Vec3(0.5, 0.5, 0.5), 0.1, 0.9, 0.0));
        
        // Lights
        add_light(Vec3(10, 10, -10), Vec3(1, 1, 1), 0.8);
        add_light(Vec3(-10, 10, -10), Vec3(1, 0.9, 0.8), 0.6);
        add_light(Vec3(0, 5, -15), Vec3(0.8, 0.8, 1), 0.4);
    }
    
    void create_simple_scene() {
        add_sphere(Vec3(0, 0, 0), 1.0, 
                  Material(Vec3(1.0, 0.3, 0.3), 0.5, 0.2, 0.0));
        add_sphere(Vec3(-2, 0, 1), 1.0, 
                  Material(Vec3(0.3, 1.0, 0.3), 0.3, 0.5, 0.0));
        add_sphere(Vec3(2, 0, 1), 1.0, 
                  Material(Vec3(0.3, 0.3, 1.0), 0.4, 0.3, 0.0));
        add_sphere(Vec3(0, -101, 0), 100, 
                  Material(Vec3(0.5, 0.5, 0.5), 0.1, 0.9, 0.0));
        add_light(Vec3(5, 5, -5), Vec3(1, 1, 1), 1.0);
    }
};

/**
 * Camera class
 */
class Camera {
public:
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    
    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, 
           double aspect_ratio, double aperture, double focus_dist) {
        double theta = vfov * M_PI / 180.0;
        double h = std::tan(theta / 2);
        double viewport_height = 2.0 * h;
        double viewport_width = aspect_ratio * viewport_height;
        
        Vec3 w = (lookfrom - lookat).normalize();
        Vec3 u = vup.cross(w).normalize();
        Vec3 v = w.cross(u);
        
        origin = lookfrom;
        horizontal = u * viewport_width * focus_dist;
        vertical = v * viewport_height * focus_dist;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w * focus_dist;
    }
    
    Ray get_ray(double s, double t) const {
        return Ray(
            origin,
            lower_left_corner + horizontal * s + vertical * t - origin
        );
    }
};

/**
 * Ray Tracer main class
 */
class RayTracer {
private:
    int width, height;
    int samples_per_pixel;
    int max_depth;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<Camera> camera;
    std::vector<Vec3> framebuffer;
    
public:
    RayTracer(int w, int h, int spp = 1, int depth = 5) 
        : width(w), height(h), samples_per_pixel(spp), max_depth(depth) {
        framebuffer.resize(width * height);
        scene = std::make_unique<Scene>();
        
        // Setup camera
        Vec3 lookfrom(13, 2, -15);
        Vec3 lookat(0, 0, 0);
        Vec3 vup(0, 1, 0);
        double vfov = 40.0;
        double aspect_ratio = double(width) / double(height);
        double aperture = 0.0;
        double focus_dist = 10.0;
        
        camera = std::make_unique<Camera>(
            lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist
        );
    }
    
    // BOTTLENECK
    inline Vec3 calculate_lighting(const Vec3& point, const Vec3& normal, 
                           const Vec3& view_dir, const Material& material) {
        Vec3 color = scene->ambient_light * material.albedo;
        
        for (const auto& light : scene->lights) {
            Vec3 light_dir = (light.position - point).normalize();
            double light_distance = (light.position - point).length();
            
            // Shadow ray
            g_counters.shadow_rays++;
            Ray shadow_ray(point + normal * 0.001, light_dir);
            double t;
            const Sphere* hit_sphere;
            bool in_shadow = scene->intersect(shadow_ray, 0.001, light_distance, t, hit_sphere);
            
            if (!in_shadow) {
                // Diffuse lighting
                double diff = std::max(0.0, normal.dot(light_dir));
                color = color + material.albedo * light.color * diff * light.intensity;
                
                // Specular
                Vec3 half_vec = (light_dir - view_dir).normalize();
                double spec = std::pow(std::max(0.0, normal.dot(half_vec)), 32);
                Vec3 spec_color = material.metallic > 0.5 ? material.albedo : Vec3(1, 1, 1);
                color = color + spec_color * spec * light.intensity * (1.0 - material.roughness);
            }
        }
        
        return color;
    }
    
    // BOTTLENECK - Recursive
    Vec3 cast_ray(const Ray& ray, int depth) {
        g_counters.total_rays++;
        
        if (depth == 0) {
            g_counters.primary_rays++;
        } else {
            g_counters.reflection_rays++;
        }
        
        if (depth >= max_depth) {
            return Vec3(0, 0, 0);
        }
        
        double t;
        const Sphere* hit_sphere;
        
        if (!scene->intersect(ray, 0.001, std::numeric_limits<double>::infinity(), t, hit_sphere)) {
            // Sky gradient
            Vec3 unit_direction = ray.direction.normalize();
            double sky_t = 0.5 * (unit_direction.y + 1.0);
            return Vec3(1.0, 1.0, 1.0) * (1.0 - sky_t) + Vec3(0.5, 0.7, 1.0) * sky_t;
        }
        
        Vec3 hit_point = ray.at(t);
        Vec3 normal = hit_sphere->normal_at(hit_point);
        Vec3 view_dir = ray.direction.normalize();
        Vec3 color = calculate_lighting(hit_point, normal, view_dir, hit_sphere->material);
        
        // Handle reflection
        if (hit_sphere->material.reflectivity > 0 && depth < max_depth) {
            Vec3 reflected = ray.direction.reflect(normal);
            Ray reflection_ray(hit_point + normal * 0.001, reflected);
            Vec3 reflection_color = cast_ray(reflection_ray, depth + 1);
            
            color = color * (1.0 - hit_sphere->material.reflectivity) + 
                   reflection_color * hit_sphere->material.reflectivity;
        }
        
        return color;
    }
    
    void render() {
        std::cout << "\n=== Ray Tracer Performance Test (PROFILING VERSION) ===" << std::endl;
        std::cout << "Resolution: " << width << "x" << height 
                  << " (" << (width * height) << " pixels)" << std::endl;
        std::cout << "Scene complexity: " << scene->spheres.size() << " spheres" << std::endl;
        std::cout << "Progress reporting: DISABLED for clean profiling" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Main rendering loop 
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                Vec3 pixel_color(0, 0, 0);
                
                for (int s = 0; s < samples_per_pixel; s++) {
                    double u = (i + 0.5) / (width - 1);
                    double v = (j + 0.5) / (height - 1);
                    
                    Ray ray = camera->get_ray(u, v);
                    pixel_color = pixel_color + cast_ray(ray, 0);
                }
                
                pixel_color = pixel_color / samples_per_pixel;
                
                // Gamma correction
                pixel_color = Vec3(
                    std::sqrt(pixel_color.x),
                    std::sqrt(pixel_color.y),
                    std::sqrt(pixel_color.z)
                );
                
                framebuffer[j * width + i] = pixel_color;
                g_counters.pixels_processed++;
            }
            
            // Simple progress indicator every 100 rows (no thread!)
            if (j % 100 == 0) {
                std::cout << "." << std::flush;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        
        std::cout << "\nRender completed in " << std::fixed << std::setprecision(3) 
                  << duration << " seconds" << std::endl;
        std::cout << "Pixels per second: " << std::fixed << std::setprecision(0) 
                  << (width * height / duration) << std::endl;
    }
    
    void save_ppm(const std::string& filename) {
        std::ofstream file(filename);
        file << "P3\n" << width << " " << height << "\n255\n";
        
        for (int j = height - 1; j >= 0; j--) {
            for (int i = 0; i < width; i++) {
                Vec3 color = framebuffer[j * width + i];
                
                int r = static_cast<int>(255.999 * std::clamp(color.x, 0.0, 1.0));
                int g = static_cast<int>(255.999 * std::clamp(color.y, 0.0, 1.0));
                int b = static_cast<int>(255.999 * std::clamp(color.z, 0.0, 1.0));
                
                file << r << " " << g << " " << b << "\n";
            }
        }
        
        file.close();
        std::cout << "Image saved to " << filename << std::endl;
    }
    
    void initialize_scene(bool complex = true) {
        if (complex) {
            scene->create_complex_scene();
        } else {
            scene->create_simple_scene();
        }
    }
};

int main(int argc, char* argv[]) {
    bool complex_scene = true;
    int width = 800;
    int height = 600;
    
    if (argc > 1 && std::string(argv[1]) == "--simple") {
        complex_scene = false;
    } else if (argc > 1 && std::string(argv[1]) == "--small") {
        width = 400;
        height = 300;
    }
    
    std::cout << "Ray Tracer - Profiling Version (no progress threads)" << std::endl;
    
    g_counters.reset();
    
    RayTracer tracer(width, height, 1, 5);
    tracer.initialize_scene(complex_scene);
    tracer.render();
    tracer.save_ppm(complex_scene ? "output_complex.ppm" : "output_simple.ppm");
    
    g_counters.print();
    
    return 0;
}