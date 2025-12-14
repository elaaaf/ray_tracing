#!/usr/bin/env python3
"""
Numba Ray Tracer with CUDA and CPU fallback support
- Uses tuple-based vectors (compatible with Numba CUDA and JIT)
- CUDA mode: One thread = one pixel (GPU acceleration)
- CPU mode: Numba JIT parallelized fallback

Run: python3 ray_tracer_numba.py
"""

import numpy as np
from numba import cuda, jit, prange
import math
import argparse
import sys
import time

# Check CUDA availability
try:
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    device_decorator = cuda.jit(device=True)
else:
    # CPU fallback - use regular jit with inline
    device_decorator = jit(inline='always')


@device_decorator
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@device_decorator
def length(v):
    return math.sqrt(dot(v, v))


@device_decorator
def normalize(v):
    l = length(v)
    if l > 0:
        return (v[0]/l, v[1]/l, v[2]/l)
    return v


@device_decorator
def vec_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


@device_decorator
def vec_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


@device_decorator
def vec_scale(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)


@device_decorator
def sphere_intersect(center, radius, orig, dir):
    oc = vec_sub(orig, center)
    a = dot(dir, dir)
    b = dot(oc, dir)
    c = dot(oc, oc) - radius*radius
    disc = b*b - a*c
    if disc < 0.0:
        return False, 0.0
    sq = math.sqrt(disc)
    t = (-b - sq) / a
    if t < 0.001:
        t = (-b + sq) / a
        if t < 0.001:
            return False, 0.0
    return True, t


@device_decorator
def shade_simple(hit_point, normal, spheres_c, spheres_r, nspheres, lights_pos, lights_col, lights_intensity, nlights):
    """Simple Blinn-Phong shading"""
    color = (0.1, 0.1, 0.1)  # ambient
    
    for li in range(nlights):
        light_dir = normalize(vec_sub(lights_pos[li], hit_point))
        diff = max(0.0, dot(normal, light_dir))
        
        # Diffuse
        color = (
            color[0] + lights_intensity[li] * diff * lights_col[li][0],
            color[1] + lights_intensity[li] * diff * lights_col[li][1],
            color[2] + lights_intensity[li] * diff * lights_col[li][2]
        )
        
        # Specular
        view_dir = normalize(hit_point)  # assuming camera at origin
        half_vec = normalize(vec_sub(light_dir, view_dir))
        spec = math.pow(max(0.0, dot(normal, half_vec)), 32.0)
        color = (
            color[0] + spec * lights_intensity[li],
            color[1] + spec * lights_intensity[li],
            color[2] + spec * lights_intensity[li]
        )
    
    return color


# CUDA kernel for GPU rendering
if CUDA_AVAILABLE:
    @cuda.jit
    def render_kernel_cuda(fb, width, height, spheres_c, spheres_r, spheres_col, nspheres, 
                      lights_pos, lights_col, lights_intensity, nlights):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return
        
        # Camera setup
        u = (x + 0.5) / (width - 1)
        v = (y + 0.5) / (height - 1)
        
        origin = (13.0, 2.0, -15.0)
        target = (0.0, 0.0, 0.0)
        
        forward = normalize(vec_sub(target, origin))
        right = normalize((forward[2], 0.0, -forward[0]))
        # Cross product for up vector
        up = normalize((
            right[1]*forward[2] - right[2]*forward[1],
            right[2]*forward[0] - right[0]*forward[2],
            right[0]*forward[1] - right[1]*forward[0]
        ))
        
        fov = 40.0 * math.pi / 180.0
        half_h = math.tan(fov / 2.0)
        aspect = float(width) / float(height)
        half_w = aspect * half_h
        
        pixel_dir = vec_add(forward, vec_scale(right, (2.0*u - 1.0)*half_w))
        pixel_dir = vec_add(pixel_dir, vec_scale(up, (2.0*v - 1.0)*half_h))
        pixel_dir = normalize(pixel_dir)
        
        # Ray casting
        closest = 1e20
        hit_any = False
        hit_normal = (0.0, 0.0, 0.0)
        hit_albedo = (0.0, 0.0, 0.0)
        
        for si in range(nspheres):
            center = (spheres_c[si][0], spheres_c[si][1], spheres_c[si][2])
            hit, t = sphere_intersect(center, spheres_r[si], origin, pixel_dir)
            if hit and t < closest:
                closest = t
                hit_any = True
                hit_point = vec_add(origin, vec_scale(pixel_dir, t))
                hit_normal = normalize(vec_scale(vec_sub(hit_point, center), 1.0/spheres_r[si]))
                hit_albedo = (spheres_col[si][0], spheres_col[si][1], spheres_col[si][2])
        
        # Shading
        if hit_any:
            hit_point = vec_add(origin, vec_scale(pixel_dir, closest))
            col = shade_simple(hit_point, hit_normal, spheres_c, spheres_r, nspheres,
                              lights_pos, lights_col, lights_intensity, nlights)
            col = (col[0]*hit_albedo[0], col[1]*hit_albedo[1], col[2]*hit_albedo[2])
        else:
            # Sky gradient
            t = 0.5 * (pixel_dir[1] + 1.0)
            col = ((1.0-t)*1.0 + t*0.5, (1.0-t)*1.0 + t*0.7, (1.0-t)*1.0 + t*1.0)
        
        # Gamma correction
        col = (math.sqrt(col[0]), math.sqrt(col[1]), math.sqrt(col[2]))
        
        idx = y * width + x
        fb[idx, 0] = col[0]
        fb[idx, 1] = col[1]
        fb[idx, 2] = col[2]


# CPU kernel for fallback rendering (using Numba JIT with parallel)
@jit(nopython=True, parallel=True)
def render_kernel_cpu(fb, width, height, spheres_c, spheres_r, spheres_col, nspheres, 
                  lights_pos, lights_col, lights_intensity, nlights):
    for y in prange(height):
        for x in range(width):
            # Camera setup
            u = (x + 0.5) / (width - 1)
            v = (y + 0.5) / (height - 1)
            
            origin = (13.0, 2.0, -15.0)
            target = (0.0, 0.0, 0.0)
            
            forward = normalize(vec_sub(target, origin))
            right = normalize((forward[2], 0.0, -forward[0]))
            # Cross product for up vector
            up = normalize((
                right[1]*forward[2] - right[2]*forward[1],
                right[2]*forward[0] - right[0]*forward[2],
                right[0]*forward[1] - right[1]*forward[0]
            ))
            
            fov = 40.0 * math.pi / 180.0
            half_h = math.tan(fov / 2.0)
            aspect = float(width) / float(height)
            half_w = aspect * half_h
            
            pixel_dir = vec_add(forward, vec_scale(right, (2.0*u - 1.0)*half_w))
            pixel_dir = vec_add(pixel_dir, vec_scale(up, (2.0*v - 1.0)*half_h))
            pixel_dir = normalize(pixel_dir)
            
            # Ray casting
            closest = 1e20
            hit_any = False
            hit_normal = (0.0, 0.0, 0.0)
            hit_albedo = (0.0, 0.0, 0.0)
            
            for si in range(nspheres):
                center = (spheres_c[si][0], spheres_c[si][1], spheres_c[si][2])
                hit, t = sphere_intersect(center, spheres_r[si], origin, pixel_dir)
                if hit and t < closest:
                    closest = t
                    hit_any = True
                    hit_point = vec_add(origin, vec_scale(pixel_dir, t))
                    hit_normal = normalize(vec_scale(vec_sub(hit_point, center), 1.0/spheres_r[si]))
                    hit_albedo = (spheres_col[si][0], spheres_col[si][1], spheres_col[si][2])
            
            # Shading
            if hit_any:
                hit_point = vec_add(origin, vec_scale(pixel_dir, closest))
                col = shade_simple(hit_point, hit_normal, spheres_c, spheres_r, nspheres,
                                  lights_pos, lights_col, lights_intensity, nlights)
                col = (col[0]*hit_albedo[0], col[1]*hit_albedo[1], col[2]*hit_albedo[2])
            else:
                # Sky gradient
                t = 0.5 * (pixel_dir[1] + 1.0)
                col = ((1.0-t)*1.0 + t*0.5, (1.0-t)*1.0 + t*0.7, (1.0-t)*1.0 + t*1.0)
            
            # Gamma correction
            col = (math.sqrt(col[0]), math.sqrt(col[1]), math.sqrt(col[2]))
            
            idx = y * width + x
            fb[idx, 0] = col[0]
            fb[idx, 1] = col[1]
            fb[idx, 2] = col[2]


def build_scene(scene_type):
    """
    Build scene based on type:
    - 'simple': 4 spheres, 1 light
    - 'complex': 5 spheres, 3 lights
    - 'extreme': 80 spheres, 3 lights (edge case for performance testing)
    """
    if scene_type == 'extreme':
        # Generate 80 spheres in a grid pattern
        spheres_list = []
        colors_list = []
        
        # Ground plane
        spheres_list.append([0, -102, 0])
        colors_list.append([0.5, 0.5, 0.5])
        
        # Generate grid of spheres (9x9 = 81 spheres total with ground)
        grid_size = 9
        spacing = 3.0
        offset = -(grid_size - 1) * spacing / 2
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = offset + i * spacing
                z = offset + j * spacing
                y = np.random.uniform(-0.5, 0.5)  # Random height variation
                
                spheres_list.append([x, y, z])
                
                # Random colors
                r = np.random.uniform(0.2, 1.0)
                g = np.random.uniform(0.2, 1.0)
                b = np.random.uniform(0.2, 1.0)
                colors_list.append([r, g, b])
        
        spheres_c = np.array(spheres_list[:80], dtype=np.float32)  # Limit to 80
        spheres_r = np.concatenate([
            np.array([100.0], dtype=np.float32),  # Ground
            np.random.uniform(0.3, 1.2, 79).astype(np.float32)  # Random radii
        ])
        spheres_col = np.array(colors_list[:80], dtype=np.float32)
        
        lights_pos = np.array([
            [15, 20, -15],
            [-15, 20, -15],
            [0, 25, 0]
        ], dtype=np.float32)
        lights_col = np.array([
            [1, 1, 1],
            [1, 0.9, 0.8],
            [0.9, 0.9, 1]
        ], dtype=np.float32)
        lights_intensity = np.array([0.7, 0.6, 0.5], dtype=np.float32)
        
    elif scene_type == 'complex':
        spheres_c = np.array([
            [0, 0, 0],
            [-4, 0, 2],
            [3, -1, 1],
            [2, 0.5, -2],
            [0, -102, 0]
        ], dtype=np.float32)
        spheres_r = np.array([2.0, 1.5, 1.0, 1.2, 100.0], dtype=np.float32)
        spheres_col = np.array([
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
            [0.8, 0.8, 0.8],
            [0.5, 0.5, 0.5]
        ], dtype=np.float32)
        lights_pos = np.array([
            [10, 10, -10],
            [-10, 10, -10],
            [0, 5, -15]
        ], dtype=np.float32)
        lights_col = np.array([
            [1, 1, 1],
            [1, 0.9, 0.8],
            [0.8, 0.8, 1]
        ], dtype=np.float32)
        lights_intensity = np.array([0.8, 0.6, 0.4], dtype=np.float32)
    else:
        spheres_c = np.array([
            [0, 0, 0],
            [-2, 0, 1],
            [2, 0, 1],
            [0, -101, 0]
        ], dtype=np.float32)
        spheres_r = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float32)
        spheres_col = np.array([
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
            [0.5, 0.5, 0.5]
        ], dtype=np.float32)
        lights_pos = np.array([[5, 5, -5]], dtype=np.float32)
        lights_col = np.array([[1, 1, 1]], dtype=np.float32)
        lights_intensity = np.array([1.0], dtype=np.float32)
    
    return spheres_c, spheres_r, spheres_col, lights_pos, lights_col, lights_intensity


def write_ppm(filename, fb, width, height):
    with open(filename, "w") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for j in range(height - 1, -1, -1):
            for i in range(width):
                c = fb[j * width + i]
                r = int(255.999 * min(max(c[0], 0.0), 1.0))
                g = int(255.999 * min(max(c[1], 0.0), 1.0))
                b = int(255.999 * min(max(c[2], 0.0), 1.0))
                f.write(f"{r} {g} {b}\n")


def main():
    parser = argparse.ArgumentParser(description='Numba Ray Tracer with CUDA/CPU support')
    parser.add_argument("--simple", action="store_true", help="Use simple scene (4 spheres)")
    parser.add_argument("--extreme", action="store_true", help="Use extreme scene (80 spheres)")
    parser.add_argument("--small", action="store_true", help="Use small resolution (400x300)")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU mode even if CUDA is available")
    parser.add_argument("--quiet", action="store_true", help="Suppress output messages")
    args = parser.parse_args()
    
    width, height = (400, 300) if args.small else (800, 600)
    
    # Determine scene type
    if args.extreme:
        scene_type = 'extreme'
    elif args.simple:
        scene_type = 'simple'
    else:
        scene_type = 'complex'
    
    # Determine execution mode
    use_cuda = CUDA_AVAILABLE and not args.force_cpu
    
    if not args.quiet:
        mode = "CUDA (GPU)" if use_cuda else "CPU (Numba JIT)"
        print(f"Ray Tracer Mode: {mode}", file=sys.stderr)
        print(f"Resolution: {width}x{height}", file=sys.stderr)
        print(f"Scene: {scene_type.capitalize()}", file=sys.stderr)
    
    spheres_c, spheres_r, spheres_col, lights_pos, lights_col, lights_intensity = build_scene(scene_type)
    nspheres = spheres_c.shape[0]
    nlights = lights_pos.shape[0]
    
    # Allocate framebuffer
    fb = np.zeros((width * height, 3), dtype=np.float32)
    
    start_time = time.time()
    
    if use_cuda:
        # CUDA mode - transfer to device
        d_fb = cuda.to_device(fb)
        d_spheres_c = cuda.to_device(spheres_c)
        d_spheres_r = cuda.to_device(spheres_r)
        d_spheres_col = cuda.to_device(spheres_col)
        d_lights_pos = cuda.to_device(lights_pos)
        d_lights_col = cuda.to_device(lights_col)
        d_lights_intensity = cuda.to_device(lights_intensity)
        
        threads = (16, 16)
        blocks = ((width + threads[0] - 1) // threads[0], 
                  (height + threads[1] - 1) // threads[1])
        
        render_kernel_cuda[blocks, threads](
            d_fb, width, height,
            d_spheres_c, d_spheres_r, d_spheres_col, nspheres,
            d_lights_pos, d_lights_col, d_lights_intensity, nlights
        )
        
        fb = d_fb.copy_to_host()
    else:
        # CPU mode - direct computation with Numba JIT
        render_kernel_cpu(
            fb, width, height,
            spheres_c, spheres_r, spheres_col, nspheres,
            lights_pos, lights_col, lights_intensity, nlights
        )
    
    elapsed_time = time.time() - start_time
    
    output_file = "output_numba.ppm"
    write_ppm(output_file, fb, width, height)
    
    if not args.quiet:
        print(f"Rendered {width}x{height} in {elapsed_time:.3f}s", file=sys.stderr)
        print(f"Output: {output_file}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
