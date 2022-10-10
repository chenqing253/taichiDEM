import taichi as ti
import math
import os

ti.init(arch=ti.gpu)
vec = ti.math.vec2

SAVE_FRAMES = False

window_size = 512 #1024   # Number of pixels of the window
n = int(8192 )  # Number of grains


coefficient = 0.75
density = 100.0
#density = 2650
stiffness = 8e3
#stiffness = 1e8
restitution_coef = 0.001
gravity = -9.81e-2
rotation = True
dt = 0.0001#0.0001  # Larger dt might lead to unstable results.
# dt_crit = sqrt(m/stiffness)
substeps = 60


@ti.dataclass
class Grain:
    id: ti.i32 # id of particle
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
    #oldfs: ti.field(dtype=ti.i32, shape=(10))

oldct = ti.field(dtype=ti.i32, shape=(n,10)) # contact in the prev 
curct = ti.field(dtype=ti.i32, shape=(n,10)) # contact in the curr 
oldfs = ti.field(dtype=ti.f32, shape=(n,10)) # shear force in prev
curfs = ti.field(dtype=ti.f32, shape=(n,10)) # shear force in curr

gf = Grain.field(shape=(n, ))

grid_n = 128
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

grain_r_min = 0.002
#grain_r_min = 0.0038
grain_r_max = 0.003
#grain_r_max = 0.0039

assert grain_r_max * 2 < grid_size
dt_crit = math.sqrt(density*math.pi*grain_r_min**2/stiffness)
print(f"critical time step : {dt_crit}")


@ti.kernel
def init():
    for i in gf:
        # assign unique id to particle 1-8192
        gf[i].id = i + 1
        # Spread grains in a restricted area.
        l = i * grid_size
        padding = 0.10
        region_width = 0.5# 1.0 - padding * 2#2
        pos = vec(l % region_width + grid_size * ti.random() * 0.20 + 0.0  ,
                 l // region_width * grid_size + grid_size * 0.5 + 0.0)
        #pos = vec(l % region_width + padding + grid_size * ti.random() * 0.2,
        #          l // region_width * grid_size + 0.0)
        gf[i].p = pos
        gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[i].m = density * math.pi * gf[i].r**2
        gf[i].o = 0.0 
        gf[i].I = 0.5 * gf[i].m * gf[i].r**2  # 1/2*m*r^2 is interia for solid disk/cylinder along z axis
        #gf[i]if (i==8192):  print(i)
  
@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a
        L = gf[i].T / gf[i].I
        #gf[i].w += (gf[i].L + L) * dt / 2.0 
        gf[i].w += ( L ) * dt 
        gf[i].o += gf[i].w * dt + 0.5 * L * dt**2
        gf[i].L = L
        #if (L!=0):177
        #if ( i==177 ):
        #    print("particle id",i,gf[i].w,gf[i].o,L)
        for k in range(10):
            oldct[i,k] = curct[i,k]
            curct[i,k] = 0
            oldfs[i,k] = curfs[i,k]
            #curfs[i,k] = 0
         

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
    # find (i,j) is exist in old contact 
    old_fs = gf[i].m - gf[i].m
    for k in range(10):
        if (oldct[i,k] == gf[j].id):
            old_fs = oldfs[i,k]

    rel_pos = gf[j].p - gf[i].p
    rel_v = gf[j].v - gf[i].v
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        # update contact 
        tmp_index = 0
        for k in range(10):
            if (curct[i,k] == 0): 
                pass 
            else:
                tmp_index += 1 
        curct[i,tmp_index] = gf[j].id
        # normal force
        normal = rel_pos / dist
        fn = delta * stiffness
        Fn = fn * normal
        # shear force: 
        tangent = vec( -normal[1], normal[0] )
        c_pos = gf[i].p + normal * ( gf[i].r - 0.5 * delta )
        cr1 = c_pos - gf[i].p 
        cr2 = c_pos - gf[j].p 
        distcr1 = ti.sqrt(cr1[0]**2 + cr1[1]**2)
        distcr2 = ti.sqrt(cr2[0]**2 + cr2[1]**2) 
        rel_vn = ( rel_v.dot(normal) ) * normal
        rel_vt = rel_v - rel_vn
        ds = ( rel_v.dot(tangent) ) # + gf[j].w * distcr2 - gf[i].w * distcr1 
        rel_ds = ds * dt
        del_fs = - 0.01 * stiffness * rel_ds 
        fs = old_fs + del_fs
        ### friction check
        max_fs = coefficient * fn
        if ( fs > max_fs ):
            fs = max_fs
        curfs[i,tmp_index] = fs
        assert fs < coefficient * fn
        Fs = fs * tangent
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        beta = ( 1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2) )
        C = 2. * beta * ti.sqrt(K * M)
        Vn = rel_v.dot(normal)
        Vs = rel_v.dot(tangent) 
        Fd = C * Vn * normal 
        if ( rotation ):
            Fd = C * Vn * normal  + 0.01*C * Vs * tangent
        # total force:
        if ( rotation ):
            gf[i].f += Fd - Fn - Fs
            gf[j].f -= Fd - Fn - Fs 
        else:
            gf[i].f += Fd - Fn 
            gf[j].f -= Fd - Fn  
         # total torque
        #if ( rotation ):
            #gf[i].T +=  cr1.cross(Fs)
            #gf[j].T -=  cr2.cross(Fs)


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
    print(meanOri)
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
