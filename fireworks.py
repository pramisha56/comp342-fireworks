import glfw
from OpenGL.GL import *
import numpy as np
import ctypes, math, time, random

# =========================
# GLSL SHADERS (embedded)
# =========================
PARTICLE_VERT = """
#version 330 core
layout (location=0) in vec2 aPos;
layout (location=1) in vec2 aUV;

layout (location=2) in vec3 iPos;
layout (location=3) in vec4 iColor;
layout (location=4) in float iSize;
layout (location=5) in float iLife;

uniform mat4 uProj;
uniform mat4 uView;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamPos;

out vec2 vUV;
out vec4 vColor;
out float vLife;
out float vDist;

void main(){
    float dist = length(iPos - uCamPos);
    float atten = 1.0 / (0.07 * dist + 1.0);

    vec3 worldPos = iPos + (uCamRight * aPos.x + uCamUp * aPos.y) * iSize * atten;

    gl_Position = uProj * uView * vec4(worldPos, 1.0);

    vUV = aUV;
    vColor = iColor;
    vLife = iLife;
    vDist = dist;
}
"""

PARTICLE_FRAG = """
#version 330 core
in vec2 vUV;
in vec4 vColor;
in float vLife;
in float vDist;

out vec4 FragColor;

uniform sampler2D uTex;
uniform float uFogStart = 45.0;
uniform float uFogEnd   = 160.0;

void main(){
    vec4 tex = texture(uTex, vUV);

    // smooth lifetime fade
    float fade = smoothstep(0.0, 0.15, vLife) * smoothstep(0.0, 1.0, vLife);

    // fog
    float fog = clamp((uFogEnd - vDist) / (uFogEnd - uFogStart), 0.0, 1.0);

    // bright HDR-ish output (additive blending)
    vec3 col = vColor.rgb * 2.3;

    float alpha = tex.a * fade * fog;
    FragColor = vec4(col * tex.r, alpha);
}
"""

SKY_VERT = """
#version 330 core
layout (location=0) in vec2 aPos;
out vec2 vPos;
void main(){
    vPos = aPos;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

SKY_FRAG = """
#version 330 core
in vec2 vPos;
out vec4 FragColor;
void main(){
    float t = (vPos.y + 1.0) * 0.5;
    vec3 top = vec3(0.01, 0.02, 0.06);
    vec3 bottom = vec3(0.00, 0.00, 0.02);
    vec3 col = mix(bottom, top, t);
    FragColor = vec4(col, 1.0);
}
"""

# =========================
# Small math helpers
# =========================
def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f / aspect
    M[1,1] = f
    M[2,2] = (far + near) / (near - far)
    M[2,3] = (2 * far * near) / (near - far)
    M[3,2] = -1.0
    return M

def look_at(eye, center, up):
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0,:3] = s
    M[1,:3] = u
    M[2,:3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3,3] = -eye
    return M @ T

# =========================
# Shader compile/link
# =========================
def compile_shader(src, shader_type):
    sh = glCreateShader(shader_type)
    glShaderSource(sh, src)
    glCompileShader(sh)
    ok = glGetShaderiv(sh, GL_COMPILE_STATUS)
    if not ok:
        msg = glGetShaderInfoLog(sh).decode()
        raise RuntimeError(f"Shader compile error:\n{msg}")
    return sh

def make_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    ok = glGetProgramiv(prog, GL_LINK_STATUS)
    if not ok:
        msg = glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error:\n{msg}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog

def set_mat4(prog, name, mat):
    loc = glGetUniformLocation(prog, name)
    glUniformMatrix4fv(loc, 1, GL_FALSE, mat.T)

    glClearColor(0.05, 0.05, 0.10, 1.0)



def set_vec3(prog, name, v):
    loc = glGetUniformLocation(prog, name)
    glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

def set_int(prog, name, i):
    loc = glGetUniformLocation(prog, name)
    glUniform1i(loc, i)

# =========================
# Glow texture generated in code (no assets)
# =========================
def make_glow_texture(size=128):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = (size-1) / 2.0
    cy = (size-1) / 2.0
    r = np.sqrt((x-cx)**2 + (y-cy)**2) / (size*0.5)
    # soft radial falloff
    alpha = np.clip(1.0 - r, 0.0, 1.0)
    alpha = alpha**2.2

    # single-channel glow into RGBA
    img = np.zeros((size, size, 4), dtype=np.uint8)
    val = (alpha * 255).astype(np.uint8)
    img[...,0] = val
    img[...,1] = val
    img[...,2] = val
    img[...,3] = val

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

# =========================
# Camera (smooth FPS style)
# =========================
class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 8.0, 35.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = -5.0
        self.fov = 55.0
        self.speed = 14.0
        self.sens = 0.08

        self.front = np.array([0,0,-1], dtype=np.float32)
        self.up = np.array([0,1,0], dtype=np.float32)
        self.right = np.array([1,0,0], dtype=np.float32)
        self.world_up = np.array([0,1,0], dtype=np.float32)

        self._first = True
        self._lastx = 0
        self._lasty = 0

        self.smooth_pos = self.pos.copy()
        self.smooth_factor = 12.0

        self.view = np.eye(4, dtype=np.float32)
        self.proj = np.eye(4, dtype=np.float32)
        self.update_vectors()

    def update_vectors(self):
        cy = math.cos(math.radians(self.yaw))
        sy = math.sin(math.radians(self.yaw))
        cp = math.cos(math.radians(self.pitch))
        sp = math.sin(math.radians(self.pitch))
        f = np.array([cy*cp, sp, sy*cp], dtype=np.float32)
        self.front = normalize(f)
        self.right = normalize(np.cross(self.front, self.world_up))
        self.up = np.cross(self.right, self.front)

    def mouse(self, window):
        x, y = glfw.get_cursor_pos(window)
        if self._first:
            self._lastx, self._lasty = x, y
            self._first = False
        dx = (x - self._lastx) * self.sens
        dy = (self._lasty - y) * self.sens
        self._lastx, self._lasty = x, y

        self.yaw += dx
        self.pitch = max(-89.0, min(89.0, self.pitch + dy))
        self.update_vectors()

    def update(self, window, dt, aspect):
        self.mouse(window)

        v = self.speed * dt
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.pos += self.front * v
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.pos -= self.front * v
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.pos -= self.right * v
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.pos += self.right * v
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            self.pos += self.world_up * v
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.pos -= self.world_up * v

        # smooth movement
        a = 1.0 - math.exp(-self.smooth_factor * dt)
        self.smooth_pos = self.smooth_pos*(1-a) + self.pos*a

        self.view = look_at(self.smooth_pos, self.smooth_pos + self.front, self.up)
        self.proj = perspective(self.fov, aspect, 0.1, 500.0)

# =========================
# Particle system (GPU instanced)
# =========================
class ParticleSystem:
    def __init__(self, max_particles=20000):
        self.max = max_particles
        self.count = 0
        self.pos = np.zeros((self.max,3), dtype=np.float32)
        self.vel = np.zeros((self.max,3), dtype=np.float32)
        self.col = np.zeros((self.max,4), dtype=np.float32)
        self.life = np.zeros((self.max,), dtype=np.float32)
        self.size = np.zeros((self.max,), dtype=np.float32)

        self.instance = np.zeros((self.max, 9), dtype=np.float32)  # pos(3)+color(4)+size+life

        self.prog = make_program(PARTICLE_VERT, PARTICLE_FRAG)
        self.tex = make_glow_texture(128)
        self._make_buffers()

    def _make_buffers(self):
        # quad vertices: aPos(xy) aUV(xy)
        quad = np.array([
            -0.5,-0.5, 0.0,0.0,
             0.5,-0.5, 1.0,0.0,
             0.5, 0.5, 1.0,1.0,
            -0.5, 0.5, 0.0,1.0
        ], dtype=np.float32)
        idx = np.array([0,1,2, 2,3,0], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

        stride = 16
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

        self.inst_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.inst_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.instance.nbytes, None, GL_STREAM_DRAW)

        inst_stride = 9 * 4
        # iPos
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)
        # iColor
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(12))
        glVertexAttribDivisor(3, 1)
        # iSize
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(28))
        glVertexAttribDivisor(4, 1)
        # iLife
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(32))
        glVertexAttribDivisor(5, 1)

        glBindVertexArray(0)

    def spawn(self, p, v, c, s, l):
        n = len(l)
        space = self.max - self.count
        n = min(n, space)
        if n <= 0:
            return
        i0, i1 = self.count, self.count + n
        self.pos[i0:i1] = p[:n]
        self.vel[i0:i1] = v[:n]
        self.col[i0:i1] = c[:n]
        self.size[i0:i1] = s[:n]
        self.life[i0:i1] = l[:n]
        self.count = i1

    def update(self, dt):
        if self.count == 0:
            return

        t = time.time()
        wind = np.array([0.35*math.sin(t*0.6), 0.0, 0.2*math.cos(t*0.4)], dtype=np.float32)
        g = np.array([0.0, -9.81, 0.0], dtype=np.float32)

        p = self.pos[:self.count]
        v = self.vel[:self.count]
        life = self.life[:self.count]

        speed = np.linalg.norm(v, axis=1) + 1e-6
        drag = 0.06 * speed
        v -= (v * drag[:,None]) * dt
        turb = (np.random.rand(self.count,3).astype(np.float32)-0.5)*0.9

        v += (g + wind) * dt + turb * dt
        p += v * dt

        life -= dt * 0.45

        alive = life > 0.0
        idx = np.where(alive)[0]
        newc = len(idx)

        self.pos[:newc] = p[idx]
        self.vel[:newc] = v[idx]
        self.col[:newc] = self.col[:self.count][idx]
        self.size[:newc] = self.size[:self.count][idx]
        self.life[:newc] = life[idx]
        self.count = newc

    def upload(self):
        if self.count == 0:
            return
        self.instance[:self.count, 0:3] = self.pos[:self.count]
        self.instance[:self.count, 3:7] = self.col[:self.count]
        self.instance[:self.count, 7] = self.size[:self.count]
        self.instance[:self.count, 8] = self.life[:self.count]

        glBindBuffer(GL_ARRAY_BUFFER, self.inst_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.instance[:self.count].nbytes, self.instance[:self.count])

    def render(self, cam: Camera):
        if self.count == 0:
            return

        self.upload()

        glUseProgram(self.prog)
        set_mat4(self.prog, "uProj", cam.proj)
        set_mat4(self.prog, "uView", cam.view)
        set_vec3(self.prog, "uCamRight", cam.right)
        set_vec3(self.prog, "uCamUp", cam.up)
        set_vec3(self.prog, "uCamPos", cam.smooth_pos)
        set_int(self.prog, "uTex", 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)

        glDepthMask(GL_FALSE)
        glBlendFunc(GL_ONE, GL_ONE)

        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.count)
        glBindVertexArray(0)

        glDepthMask(GL_TRUE)

# =========================
# Fireworks logic (few types, still single-file)
# =========================
def hsv_to_rgb(h, s, v):
    i = int(h*6); f = h*6-i
    p = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s)
    i %= 6
    if i==0: return v,t,p
    if i==1: return q,v,p
    if i==2: return p,v,t
    if i==3: return p,q,v
    if i==4: return t,p,v
    return v,p,q

class Fireworks:
    def __init__(self):
        self.ps = ParticleSystem(22000)
        self.rockets = []  # pos, vel, fuse, type, stage
        self.last = 0.0
        self.interval = 1.2
        self.auto = True

    def launch(self, pos=None, ftype=None):
        if pos is None:
            pos = np.array([random.uniform(-18,18), 0.0, random.uniform(-18,18)], dtype=np.float32)
        if ftype is None:
            ftype = random.choice(["burst","ring","heart","spiral","double","crackle"])
        vel = np.array([random.uniform(-1,1), random.uniform(18,26), random.uniform(-1,1)], dtype=np.float32)
        fuse = random.uniform(1.4, 2.1)
        self.rockets.append([pos, vel, fuse, ftype, 0])

    def update(self, dt):
        now = time.time()
        if self.auto and (now - self.last) > self.interval:
            self.launch()
            self.last = now

        g = np.array([0,-9.81,0], dtype=np.float32)
        wind = np.array([0.25*math.sin(now*0.5), 0.0, 0.2*math.cos(now*0.3)], dtype=np.float32)

        newr = []
        for pos, vel, fuse, ftype, stage in self.rockets:
            vel += (g + wind) * dt
            pos += vel * dt
            fuse -= dt

            # rocket trail
            n = 6
            tp = np.repeat(pos[None,:], n, axis=0)
            tv = (np.random.rand(n,3).astype(np.float32)-0.5)*1.2
            tc = np.zeros((n,4), dtype=np.float32)
            tc[:,:3] = np.array([1.0,0.8,0.3])
            tc[:,3] = 1.0
            ts = np.full((n,), 7.0, dtype=np.float32)
            tl = np.full((n,), 0.35, dtype=np.float32)
            self.ps.spawn(tp,tv,tc,ts,tl)

            if fuse <= 0 or vel[1] < 0:
                self.explode(pos, ftype, stage)
            else:
                newr.append([pos, vel, fuse, ftype, stage])
        self.rockets = newr

        self.ps.update(dt)

    def explode(self, pos, ftype, stage):
        base_h = random.random()
        if ftype == "heart":
            self._heart(pos)
        elif ftype == "spiral":
            self._spiral(pos, base_h)
        elif ftype == "double":
            if stage == 0:
                self._burst(pos, base_h, 200, (7,12))
                self.rockets.append([pos.copy(), np.array([0, random.uniform(5,9), 0],dtype=np.float32),
                                     random.uniform(0.35,0.6), "burst", 1])
            else:
                self._burst(pos, (base_h+0.2)%1.0, 260, (9,15))
        elif ftype == "crackle":
            self._burst(pos, base_h, 160, (7,12))
            self._crackle(pos)
        elif ftype == "ring":
            self._ring(pos, base_h)
        else:
            self._burst(pos, base_h, 220, (8,14))

    def _burst(self, pos, base_h, n, speed_rng):
        theta = np.random.rand(n).astype(np.float32) * (2*np.pi)
        u = np.random.rand(n).astype(np.float32) * 2 - 1
        phi = np.arccos(u)

        sp = (speed_rng[0] + (speed_rng[1]-speed_rng[0])*np.random.rand(n)).astype(np.float32)

        vx = sp*np.sin(phi)*np.cos(theta)
        vy = sp*np.cos(phi)
        vz = sp*np.sin(phi)*np.sin(theta)
        v = np.stack([vx,vy,vz], axis=1).astype(np.float32)

        p = np.repeat(pos[None,:], n, axis=0).astype(np.float32)

        c = np.zeros((n,4), dtype=np.float32)
        for i in range(n):
            r,g,b = hsv_to_rgb((base_h + random.uniform(-0.08,0.08))%1.0, 0.9, 1.0)
            c[i,:3] = (r,g,b)
        c[:,3]=1.0

        s = (10 + 8*np.random.rand(n)).astype(np.float32)
        l = (1.0 + 0.3*np.random.rand(n)).astype(np.float32)
        self.ps.spawn(p,v,c,s,l)

    def _ring(self, pos, base_h, n=220):
        ang = np.linspace(0, 2*np.pi, n, endpoint=False).astype(np.float32)
        sp = (10 + 4*np.random.rand(n)).astype(np.float32)
        vx = sp*np.cos(ang)
        vz = sp*np.sin(ang)
        vy = (np.random.rand(n).astype(np.float32)-0.5)*2.0
        v = np.stack([vx,vy,vz], axis=1).astype(np.float32)

        p = np.repeat(pos[None,:], n, axis=0).astype(np.float32)

        c = np.zeros((n,4), dtype=np.float32)
        for i in range(n):
            r,g,b = hsv_to_rgb((base_h + i/n*0.2)%1.0, 0.9, 1.0)
            c[i,:3] = (r,g,b)
        c[:,3]=1.0

        s = np.full((n,), 12.0, dtype=np.float32)
        l = np.full((n,), 1.1, dtype=np.float32)
        self.ps.spawn(p,v,c,s,l)

    def _heart(self, pos, n=240):
        t = (np.random.rand(n).astype(np.float32) * 2*np.pi)
        x = 16*np.sin(t)**3
        y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

        dir2 = np.stack([x,y], axis=1)
        dir2 /= (np.linalg.norm(dir2, axis=1)[:,None] + 1e-6)

        sp = (9 + 6*np.random.rand(n)).astype(np.float32)
        vx = dir2[:,0]*sp*0.9
        vy = dir2[:,1]*sp*0.9
        vz = (np.random.rand(n).astype(np.float32)-0.5)*3.0
        v = np.stack([vx,vy,vz], axis=1).astype(np.float32)

        p = np.repeat(pos[None,:], n, axis=0).astype(np.float32)
        c = np.zeros((n,4), dtype=np.float32)
        c[:,:3] = np.array([1.0,0.35,0.55])  # pink
        c[:,3]=1.0
        s = (12 + 8*np.random.rand(n)).astype(np.float32)
        l = np.full((n,), 1.2, dtype=np.float32)
        self.ps.spawn(p,v,c,s,l)

    def _spiral(self, pos, base_h, n=260):
        t = np.linspace(0, 6*np.pi, n).astype(np.float32)
        r = np.linspace(0.25, 1.0, n).astype(np.float32)
        vx = np.cos(t)*r
        vz = np.sin(t)*r
        vy = (np.random.rand(n).astype(np.float32)*0.6 + 0.2)
        d = np.stack([vx,vy,vz], axis=1)
        d /= (np.linalg.norm(d, axis=1)[:,None] + 1e-6)

        sp = (10 + 5*np.random.rand(n)).astype(np.float32)
        v = d * sp[:,None]

        p = np.repeat(pos[None,:], n, axis=0).astype(np.float32)
        c = np.zeros((n,4), dtype=np.float32)
        for i in range(n):
            rr,gg,bb = hsv_to_rgb((base_h + i/n*0.35)%1.0, 0.9, 1.0)
            c[i,:3] = (rr,gg,bb)
        c[:,3]=1.0
        s = (10 + 7*np.random.rand(n)).astype(np.float32)
        l = np.full((n,), 1.15, dtype=np.float32)
        self.ps.spawn(p,v,c,s,l)

    def _crackle(self, pos, n=120):
        ang = np.random.rand(n).astype(np.float32) * 2*np.pi
        sp = (14 + 10*np.random.rand(n)).astype(np.float32)
        vx = np.cos(ang)*sp
        vz = np.sin(ang)*sp
        vy = (np.random.rand(n).astype(np.float32)*6.0)
        v = np.stack([vx,vy,vz], axis=1).astype(np.float32)

        p = np.repeat(pos[None,:], n, axis=0).astype(np.float32)
        c = np.zeros((n,4), dtype=np.float32)
        c[:,:3] = np.array([1.0,0.75,0.25])  # gold
        c[:,3]=1.0
        s = np.full((n,), 7.0, dtype=np.float32)
        l = (0.35 + 0.25*np.random.rand(n)).astype(np.float32)
        self.ps.spawn(p,v,c,s,l)

    def render(self, cam):
        self.ps.render(cam)

# =========================
# Sky renderer (fullscreen quad)
# =========================
class Sky:
    def __init__(self):
        self.prog = make_program(SKY_VERT, SKY_FRAG)
        quad = np.array([-1,-1,  1,-1,  1,1,  -1,1], dtype=np.float32)
        idx  = np.array([0,1,2, 2,3,0], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,8,ctypes.c_void_p(0))

        glBindVertexArray(0)

    def render(self):
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.prog)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

# =========================
# Main
# =========================
def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    # Core profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    W, H = 1280, 720
    window = glfw.create_window(W, H, "Modern Fireworks (Single File)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glViewport(0,0,W,H)
    glEnable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)

    cam = Camera()
    sky = Sky()
    fw = Fireworks()

    last = time.time()
    fps_timer = time.time()
    frames = 0
    fps = 0.0

    while not glfw.window_should_close(window):
        now = time.time()
        dt = now - last
        last = now

        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # toggle auto
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            fw.auto = True

        cam.update(window, dt, W/H)
        fw.update(dt)

        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        sky.render()
        fw.render(cam)

        glfw.swap_buffers(window)

        # window title as UI
        frames += 1
        if now - fps_timer > 0.5:
            fps = frames / (now - fps_timer)
            frames = 0
            fps_timer = now
            glfw.set_window_title(window, f"Modern Fireworks | FPS: {fps:.1f} | Particles: {fw.ps.count}")

    glfw.terminate()

if __name__ == "__main__":
    print("Controls: Mouse Look, WASD, Space/Shift, ESC exit")
    print("Auto fireworks on. (You can extend toggles easily.)")
    main()
