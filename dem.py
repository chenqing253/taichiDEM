import taichi as ti
import math
import os

ti.init(arch=ti.gpu)
vec = ti.math.vec2

SAVE_FRAMES = False

window_size = 512 #1024   # Number of pixels of the window
n = 8192  # Number of grains

coefficient = 0.50
density = 100.0
density = 2650e3
stiffness = 8e3
stiffness = 1e8
restitution_coef = 0.001
gravity = -9.81e-00
rotation = True
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60


@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force
    o: ti.f32  # orientation
    I: ti.f32  # inertia
    # only magnitude, sign represent direction in z
    w: ti.f32  # angular velocity
    L: ti.f32  # angular acceleration, emm, though usualaly we use alpha
    T: ti.f32  # torque



gf = Grain.field(shape=(n, ))

grid_n = 128
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

grain_r_min = 0.002
#grain_r_min = 0.0039
grain_r_max = 0.003
#grain_r_max = 0.0039

assert grain_r_max * 2 < grid_size


@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        l = i * grid_size
        padding = 0.10
        region_width = 1.0 - padding * 2#2
        pos = vec(l % region_width + padding + grid_size * ti.random() * 0.2,
                 l // region_width * grid_size + 0.0)
        #pos = vec(l % region_width + 0.0 + grid_size * ti.random() * 0.2,
        #          l // region_width * grid_size + 0.0)
        gf[i].p = pos
        gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[i].m = density * math.pi * gf[i].r**2
        gf[i].o = 0.0 
        gf[i].I = 0.5 * gf[i].m * gf[i].r**2  # 1/2*m*r^2 is interia for solid disk/cylinder along z axis


@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a
        L = gf[i].T / gf[i].I
        gf[i].w += (gf[i].L + L) * dt / 2.0 
        gf[i].o += gf[i].w * dt + 0.5 * L * dt**2
        gf[i].L = L


@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0]
        y = gf[i].p[1]

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > 1.0:
            gf[i].p[1] = 1.0 - gf[i].r
            gf[i].v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > 1.0:
            gf[i].p[0] = 1.0 - gf[i].r
            gf[i].v[0] *= -bounce_coef


@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        # normal force
        normal = rel_pos / dist
        fn = normal * delta * stiffness
        # shear force: 
        c_pos = gf[i].p + normal * ( gf[i].r - 0.5 * delta )
        cr1 = c_pos - gf[i].p 
        cr2 = c_pos - gf[j].p 
        distcr1 = ti.sqrt(cr1[0]**2 + cr1[1]**2)
        distcr2 = ti.sqrt(cr2[0]**2 + cr2[1]**2)
        rel_v = (gf[j].v - gf[i].v) 
        rel_n = (rel_v * normal) * normal
        rel_t = rel_v - rel_n
        tangent = vec( -normal[1], normal[0] )
        vs = ( rel_v * tangent ) - gf[j].w * distcr2 - gf[i].w * distcr1 
        delta_s = vs * dt
        ## notice , fs should calculate in a cumulate way, 
        ## i.e. the fs would be record in the contact (if the contact still exist)
        ## fs -= stiffness * delta_s * tangent
        fs = - stiffness * delta_s * tangent
        ### friction check
        mag_fs = ti.sqrt(fs[0]**2 + fs[1]**2)
        max_fs = coefficient * delta * stiffness
        if ( mag_fs > max_fs ):
            fs = fs * max_fs / mag_fs
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        Vs = (gf[j].v - gf[i].v) * tangent
        fd = C * V * normal
        #if (rotation):
        #    fd = C * V * normal + C * Vs * tangent
        # total force:
        fsum = fn
        if (rotation):
            fsum = fn + fs
        gf[i].f += fd - fsum
        gf[j].f -= fd - fsum
        # total torque
        if (rotation):
            gf[i].T += - cr1.cross(fs) 
            gf[j].T -= - cr2.cross(fs) 
            #gf[i].T += cr1.cross(fd) - cr1.cross(fs) 
            #gf[j].T -= cr2.cross(fd) - cr2.cross(fs) 
        


list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        grain_count[grid_idx] += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    prefix_sum[0, 0] = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

    # Brute-force collision detection
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                   list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i < j:
                        resolve(i, j)


init()
gui = ti.GUI('Taichi DEM', (window_size, window_size))
step = 0

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)

while gui.running:
    for s in range(substeps):
        update()
        apply_bc()
        contact(gf)
    pos = gf.p.to_numpy()
    r = gf.r.to_numpy() * window_size
    
    import numpy as np
    ori = gf.o.to_numpy()
    indices = np.zeros(n,dtype=int)
    meanOri = np.mean(abs(ori))
    #print(meanOri)
    for i in range(0,len(indices)):
        if (ori[i] > (meanOri)):
            indices[i]=1
        elif (ori[i] < -meanOri ):
            indices[i]=2
    colors = np.array([ 0xEEEEF0, 0xED553B, 0x068587], dtype=np.uint32) 

    #gui.circles( pos, radius=r, palette=colors,palette_indices=indices )
    gui.circles( pos, radius=r, color=colors[indices] )
    if SAVE_FRAMES:
        gui.show(f'output/{step:06d}.png')
    else:
        gui.show()
    step += 1
