"""
Minimal CUDA Python (Numba) Ray Tracer
- Uses tuple-based vectors (more compatible with Numba CUDA)
- One thread = one pixel

Run: python3 ray_tracer_cuda_python.py
"""

import numpy as np
from numba import cuda
import math
import argparse


@cuda.jit(device=True)
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@cuda.jit(device=True)
def length(v):
    return math.sqrt(dot(v, v))


@cuda.jit(device=True)
def normalize(v):
    l = length(v)
    if l > 0:
        return (v[0]/l, v[1]/l, v[2]/l)
    return v


@cuda.jit(device=True)
def vec_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


@cuda.jit(device=True)
def vec_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


@cuda.jit(device=True)
def vec_scale(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)


@cuda.jit(device=True)
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


@cuda.jit(device=True)
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


@cuda.jit
def render_kernel(fb, width, height, spheres_c, spheres_r, spheres_col, nspheres, 
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


def build_scene(complex_scene):
    if complex_scene:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    
    width, height = (400, 300) if args.small else (800, 600)
    complex_scene = not args.simple
    
    spheres_c, spheres_r, spheres_col, lights_pos, lights_col, lights_intensity = build_scene(complex_scene)
    nspheres = spheres_c.shape[0]
    nlights = lights_pos.shape[0]
    
    # Allocate and transfer to device
    fb = np.zeros((width * height, 3), dtype=np.float32)
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
    
    render_kernel[blocks, threads](
        d_fb, width, height,
        d_spheres_c, d_spheres_r, d_spheres_col, nspheres,
        d_lights_pos, d_lights_col, d_lights_intensity, nlights
    )
    
    fb = d_fb.copy_to_host()
    write_ppm("output_cuda_py.ppm", fb, width, height)
    print(f"Rendered {width}x{height} with CUDA Python (Numba)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
