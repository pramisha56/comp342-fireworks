import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random
import math
import time
from dataclasses import dataclass
from typing import List


class TransformationMatrix:
    @staticmethod
    def identity():
        """
        Identity Matrix - no transformation
        [1 0 0 0]
        [0 1 0 0]
        [0 0 1 0]
        [0 0 0 1]
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def translation(tx, ty, tz):
        """
        Translation Matrix
        
        Mathematical form:
        [1  0  0  tx]   [x]   [x + tx]
        [0  1  0  ty] × [y] = [y + ty]
        [0  0  1  tz]   [z]   [z + tz]
        [0  0  0  1 ]   [1]   [  1   ]
        """
        return np.array([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_x(angle_degrees):
        """
        Rotation around X-axis
        
        Mathematical form:
        [1    0         0      0]
        [0  cos(θ)  -sin(θ)   0]
        [0  sin(θ)   cos(θ)   0]
        [0    0         0      1]
        """
        theta = math.radians(angle_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        return np.array([
            [1.0, 0.0,        0.0,         0.0],
            [0.0, cos_theta, -sin_theta,   0.0],
            [0.0, sin_theta,  cos_theta,   0.0],
            [0.0, 0.0,        0.0,         1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_y(angle_degrees):
        """
        Rotation around Y-axis
        
        Mathematical form:
        [ cos(θ)   0  sin(θ)  0]
        [   0      1    0     0]
        [-sin(θ)   0  cos(θ)  0]
        [   0      0    0     1]
        """
        theta = math.radians(angle_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        return np.array([
            [ cos_theta, 0.0, sin_theta, 0.0],
            [ 0.0,       1.0, 0.0,       0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [ 0.0,       0.0, 0.0,       1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_z(angle_degrees):
        """
        Rotation around Z-axis
        
        Mathematical form:
        [cos(θ)  -sin(θ)  0  0]
        [sin(θ)   cos(θ)  0  0]
        [  0        0     1  0]
        [  0        0     0  1]
        """
        theta = math.radians(angle_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        return np.array([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta,  cos_theta, 0.0, 0.0],
            [0.0,        0.0,       1.0, 0.0],
            [0.0,        0.0,       0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_arbitrary(angle_degrees, axis):
        """
        Rotation around arbitrary axis
        Uses Rodrigues' rotation formula. Something new, not in our course.
        
        axis: normalized 3D vector [x, y, z]
        """
        theta = math.radians(angle_degrees)
        
        # Normalize axis
        axis_length = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
        if axis_length == 0:
            return TransformationMatrix.identity()
        
        ux = axis[0] / axis_length
        uy = axis[1] / axis_length
        uz = axis[2] / axis_length
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        one_minus_cos = 1.0 - cos_theta
        
        # Rodrigues' formula components
        return np.array([
            [
                cos_theta + ux*ux*one_minus_cos,
                ux*uy*one_minus_cos - uz*sin_theta,
                ux*uz*one_minus_cos + uy*sin_theta,
                0.0
            ],
            [
                uy*ux*one_minus_cos + uz*sin_theta,
                cos_theta + uy*uy*one_minus_cos,
                uy*uz*one_minus_cos - ux*sin_theta,
                0.0
            ],
            [
                uz*ux*one_minus_cos - uy*sin_theta,
                uz*uy*one_minus_cos + ux*sin_theta,
                cos_theta + uz*uz*one_minus_cos,
                0.0
            ],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def scaling(sx, sy, sz):
        """
        Scaling Matrix
        Scales by factors (sx, sy, sz)
        
        Mathematical form:
        [sx  0   0  0]   [x]   [sx·x]
        [0  sy   0  0] × [y] = [sy·y]
        [0   0  sz  0]   [z]   [sz·z]
        [0   0   0  1]   [1]   [ 1  ]
        """
        return np.array([
            [sx,  0.0, 0.0, 0.0],
            [0.0, sy,  0.0, 0.0],
            [0.0, 0.0, sz,  0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def shearing_xy(shx, shy):
        """
        Shearing in XY plane
        shx: shear X based on Y
        shy: shear Y based on X
        
        Mathematical form:
        [1   shx  0  0]   [x]   [x + shx·y]
        [shy  1   0  0] × [y] = [y + shy·x]
        [0    0   1  0]   [z]   [    z    ]
        [0    0   0  1]   [1]   [    1    ]
        """
        return np.array([
            [1.0, shx, 0.0, 0.0],
            [shy, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def shearing_xz(shx, shz):
        """
        Shearing in XZ plane
        shx: shear X based on Z
        shz: shear Z based on X
        """
        return np.array([
            [1.0, 0.0, shx, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [shz, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def shearing_yz(shy, shz):
        """
        Shearing in YZ plane
        shy: shear Y based on Z
        shz: shear Z based on Y
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, shy, 0.0],
            [0.0, shz, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def apply_to_point(matrix, point):
        """
        Apply 4x4 transformation matrix to a 3D point
        
        Process:
        1. Convert point to homogeneous coordinates [x, y, z, 1]
        2. Multiply by transformation matrix
        3. Convert back to 3D coordinates
        
        Returns: Transformed 3D point as numpy array
        """
        # Convert to homogeneous coordinates
        homogeneous_point = np.array([
            point[0],
            point[1],
            point[2],
            1.0
        ], dtype=np.float32)
        
        # Matrix multiplication: M × P
        transformed = np.dot(matrix, homogeneous_point)
        
        # Convert back to 3D (perspective divide if w != 1)
        if transformed[3] != 0.0:
            return np.array([
                transformed[0] / transformed[3],
                transformed[1] / transformed[3],
                transformed[2] / transformed[3]
            ], dtype=np.float32)
        
        return np.array([
            transformed[0],
            transformed[1],
            transformed[2]
        ], dtype=np.float32)
    
    @staticmethod
    def compose(*matrices):
        """
        Compose multiple transformation matrices
        Matrices are applied right-to-left (like function composition)
        
        Example:
            M = compose(M1, M2, M3)
            Applies: M3, then M2, then M1
        """
        result = TransformationMatrix.identity()
        
        # Apply in reverse order (right-to-left)
        for matrix in reversed(matrices):
            result = np.dot(result, matrix)
        
        return result


@dataclass
class Particle:
    """Particle with position, velocity, color, and life"""
    position: np.ndarray
    velocity: np.ndarray
    color: np.ndarray
    life: float
    size: float
    trail: list


class Firework:
    """Firework with rocket and explosion particles"""
    
    def __init__(self, position, firework_type="burst"):
        self.position = position.copy()
        self.particles = []
        self.exploded = False
        self.rocket = None
        self.type = firework_type
        self.age = 0.0
        
        self.launch_rocket()
    
    def launch_rocket(self):
        """Launch rocket particle"""
        velocity = np.array([
            random.uniform(-0.5, 0.5),
            random.uniform(18.0, 25.0),
            random.uniform(-0.5, 0.5)
        ], dtype=np.float32)
        
        self.rocket = Particle(
            position=self.position.copy(),
            velocity=velocity,
            color=np.array([1.0, 1.0, 0.8, 1.0], dtype=np.float32),
            life=1.0,
            size=3.0,
            trail=[]
        )
    
    def explode(self):
        """Create explosion using transformation matrices"""
        if self.exploded:
            return
        
        self.exploded = True
        explosion_position = self.rocket.position.copy()
        
        # Number of particles
        num_particles = random.randint(150, 250)
        
        # Random color
        base_hue = random.random()
        
        # Random rotation for variety (ROTATION TRANSFORMATION)
        rotation_angle = random.uniform(0, 360)
        rotation_axis = np.array([
            random.uniform(-1, 1),
            random.uniform(0.5, 1),
            random.uniform(-1, 1)
        ], dtype=np.float32)
        
        for i in range(num_particles):
            
            # Generate base velocity based on type
            if self.type == "burst":
                # Spherical burst
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi)
                speed = random.uniform(10.0, 16.0)
                
                base_velocity = np.array([
                    speed * math.sin(phi) * math.cos(theta),
                    speed * math.cos(phi),
                    speed * math.sin(phi) * math.sin(theta)
                ], dtype=np.float32)
                
            elif self.type == "ring":
                # Circular ring (SCALING TRANSFORMATION - flatten Y)
                angle = (i / num_particles) * 2 * math.pi
                speed = random.uniform(12.0, 18.0)
                
                base_velocity = np.array([
                    speed * math.cos(angle),
                    random.uniform(-1.0, 1.0),
                    speed * math.sin(angle)
                ], dtype=np.float32)
                
            elif self.type == "fountain":
                # Fountain with shearing (SHEARING TRANSFORMATION)
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(8.0, 14.0)
                vertical_bias = random.uniform(0.6, 0.9)
                
                base_velocity = np.array([
                    speed * math.cos(angle) * (1 - vertical_bias),
                    speed * vertical_bias,
                    speed * math.sin(angle) * (1 - vertical_bias)
                ], dtype=np.float32)
                
            elif self.type == "willow":
                # Willow droop effect
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi / 2)
                speed = random.uniform(7.0, 12.0)
                
                base_velocity = np.array([
                    speed * math.sin(phi) * math.cos(theta),
                    speed * math.cos(phi) * 0.4,
                    speed * math.sin(phi) * math.sin(theta)
                ], dtype=np.float32)
                
            else:
                base_velocity = np.array([
                    random.uniform(-12.0, 12.0),
                    random.uniform(-12.0, 12.0),
                    random.uniform(-12.0, 12.0)
                ], dtype=np.float32)
            
            # APPLY ROTATION TRANSFORMATION
            R = TransformationMatrix.rotation_arbitrary(rotation_angle, rotation_axis)
            rotated_velocity = TransformationMatrix.apply_to_point(R, base_velocity)
            
            # APPLY SCALING TRANSFORMATION
            scale_variation = random.uniform(0.85, 1.15)
            S = TransformationMatrix.scaling(scale_variation, scale_variation, scale_variation)
            scaled_velocity = TransformationMatrix.apply_to_point(S, rotated_velocity)
            
            # APPLY SHEARING TRANSFORMATION (for certain types)
            if self.type in ["fountain", "willow"]:
                shear_factor = random.uniform(-0.4, 0.4)
                SH = TransformationMatrix.shearing_xy(shear_factor, 0.0)
                final_velocity = TransformationMatrix.apply_to_point(SH, scaled_velocity)
            else:
                final_velocity = scaled_velocity
            
            # Generate color with variation
            hue_offset = random.uniform(-0.1, 0.1)
            final_hue = (base_hue + hue_offset) % 1.0
            
            r, g, b = self.hsv_to_rgb(final_hue, 0.9, 1.0)
            
            particle = Particle(
                position=explosion_position.copy(),
                velocity=final_velocity,
                color=np.array([r, g, b, 1.0], dtype=np.float32),
                life=1.0,
                size=random.uniform(2.5, 4.5),
                trail=[]
            )
            
            self.particles.append(particle)
    
    @staticmethod
    def hsv_to_rgb(h, s, v):
        """Convert HSV to RGB"""
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        
        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        if i == 5: return v, p, q
        return v, v, v
    
    def update(self, dt):
        """Update firework state"""
        self.age += dt
        
        gravity = np.array([0.0, -9.8, 0.0], dtype=np.float32)
        
        # Update rocket
        if not self.exploded and self.rocket:
            # TRANSLATION: Update rocket position using transformation matrix
            displacement = self.rocket.velocity * dt
            T = TransformationMatrix.translation(displacement[0], displacement[1], displacement[2])
            self.rocket.position = TransformationMatrix.apply_to_point(T, self.rocket.position)
            
            # Update velocity
            self.rocket.velocity = self.rocket.velocity + gravity * dt
            
            # Trail
            if len(self.rocket.trail) < 20:
                self.rocket.trail.append(self.rocket.position.copy())
            
            # Decay
            self.rocket.life -= dt * 0.5
            self.rocket.color[3] = self.rocket.life
            
            # Explode when falling or life runs out
            if self.rocket.velocity[1] < 0 or self.rocket.life <= 0:
                self.explode()
        
        # Update particles
        if self.exploded:
            alive_particles = []
            
            for particle in self.particles:
                # TRANSLATION: Update particle position using transformation matrix
                displacement = particle.velocity * dt
                T = TransformationMatrix.translation(displacement[0], displacement[1], displacement[2])
                particle.position = TransformationMatrix.apply_to_point(T, particle.position)
                
                # SCALING: Apply drag to velocity (velocity scaling)
                drag_factor = 0.98
                S = TransformationMatrix.scaling(drag_factor, drag_factor, drag_factor)
                particle.velocity = TransformationMatrix.apply_to_point(S, particle.velocity)
                
                # Update velocity with gravity
                particle.velocity = particle.velocity + gravity * dt
                
                # Trail
                if len(particle.trail) < 10:
                    particle.trail.append(particle.position.copy())
                else:
                    particle.trail.pop(0)
                    particle.trail.append(particle.position.copy())
                
                # Life decay
                particle.life -= dt * 0.5
                particle.color[3] = max(0.0, particle.life)
                
                if particle.life > 0:
                    alive_particles.append(particle)
            
            self.particles = alive_particles
            
            return len(self.particles) > 0
        
        return True
    
    def render(self):
        """Render firework"""
        # Render rocket trail
        if self.rocket and not self.exploded:
            if len(self.rocket.trail) > 1:
                glBegin(GL_LINE_STRIP)
                for i, pos in enumerate(self.rocket.trail):
                    alpha = (i / len(self.rocket.trail)) * self.rocket.color[3]
                    glColor4f(1.0, 1.0, 0.5, alpha * 0.5)
                    glVertex3f(pos[0], pos[1], pos[2])
                glEnd()
            
            # Render rocket
            glPointSize(self.rocket.size)
            glBegin(GL_POINTS)
            glColor4f(self.rocket.color[0], self.rocket.color[1], 
                     self.rocket.color[2], self.rocket.color[3])
            glVertex3f(self.rocket.position[0], self.rocket.position[1], self.rocket.position[2])
            glEnd()
        
        # Render particles
        for particle in self.particles:
            # Trail
            if len(particle.trail) > 1:
                glBegin(GL_LINE_STRIP)
                for i, pos in enumerate(particle.trail):
                    alpha = (i / len(particle.trail)) * particle.color[3] * 0.3
                    glColor4f(particle.color[0], particle.color[1], particle.color[2], alpha)
                    glVertex3f(pos[0], pos[1], pos[2])
                glEnd()
            
            # Glow (outer)
            glPointSize(particle.size * 2.5)
            glBegin(GL_POINTS)
            glColor4f(particle.color[0], particle.color[1], particle.color[2], particle.color[3] * 0.2)
            glVertex3f(particle.position[0], particle.position[1], particle.position[2])
            glEnd()
            
            # Core
            glPointSize(particle.size)
            glBegin(GL_POINTS)
            glColor4f(particle.color[0], particle.color[1], particle.color[2], particle.color[3])
            glVertex3f(particle.position[0], particle.position[1], particle.position[2])
            glEnd()


# ============================================================================
# CAMERA SYSTEM
# ============================================================================

class Camera:
    """First-person camera with mouse look"""
    
    def __init__(self):
        self.position = np.array([0.0, 8.0, 35.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 18.0
        self.sensitivity = 0.1
        self.fov = 50.0
        
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        self.last_x = 400
        self.last_y = 300
        self.first_mouse = True
        
        self.update_vectors()
    
    def update_vectors(self):
        """Update camera direction vectors"""
        front_x = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front_y = math.sin(math.radians(self.pitch))
        front_z = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        
        front_length = math.sqrt(front_x**2 + front_y**2 + front_z**2)
        self.front = np.array([front_x/front_length, front_y/front_length, front_z/front_length], dtype=np.float32)
        
        # Calculate right vector
        right_x = self.front[1] * 0.0 - self.front[2] * 1.0
        right_y = self.front[2] * 0.0 - self.front[0] * 0.0
        right_z = self.front[0] * 1.0 - self.front[1] * 0.0
        right_length = math.sqrt(right_x**2 + right_y**2 + right_z**2)
        self.right = np.array([right_x/right_length, right_y/right_length, right_z/right_length], dtype=np.float32)
        
        # Calculate up vector
        up_x = self.right[1] * self.front[2] - self.right[2] * self.front[1]
        up_y = self.right[2] * self.front[0] - self.right[0] * self.front[2]
        up_z = self.right[0] * self.front[1] - self.right[1] * self.front[0]
        self.up = np.array([up_x, up_y, up_z], dtype=np.float32)
    
    def process_mouse(self, xpos, ypos):
        """Handle mouse movement"""
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
        
        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos
        
        xoffset *= self.sensitivity
        yoffset *= self.sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0
        
        self.update_vectors()
    
    def process_scroll(self, yoffset):
        """Handle scroll for zoom"""
        self.fov -= yoffset * 2
        if self.fov < 20.0:
            self.fov = 20.0
        if self.fov > 80.0:
            self.fov = 80.0
    
    def get_view_matrix(self):
        """Get view matrix components"""
        center = self.position + self.front
        return self.position, center, self.up


# ============================================================================
# MAIN SIMULATION
# ============================================================================

class FireworkSimulation:
    """Main simulation controller"""
    
    def __init__(self):
        self.fireworks = []
        self.camera = Camera()
        self.auto_launch = True
        self.last_launch = 0
        self.launch_interval = 1.8
        
        self.firework_types = ["burst", "ring", "fountain", "willow"]
        self.current_type = 0
    
    def add_firework(self, position=None, fw_type=None):
        """Add new firework"""
        if position is None:
            position = np.array([
                random.uniform(-20.0, 20.0),
                0.0,
                random.uniform(-20.0, 20.0)
            ], dtype=np.float32)
        
        if fw_type is None:
            fw_type = random.choice(self.firework_types)
        
        self.fireworks.append(Firework(position, fw_type))
    
    def cycle_type(self):
        """Cycle through firework types"""
        self.current_type = (self.current_type + 1) % len(self.firework_types)
        print(f"Firework type: {self.firework_types[self.current_type]}")
    
    def update(self, dt):
        """Update simulation"""
        current_time = time.time()
        
        if self.auto_launch and current_time - self.last_launch > self.launch_interval:
            self.add_firework()
            self.last_launch = current_time
        
        alive = []
        for fw in self.fireworks:
            if fw.update(dt):
                alive.append(fw)
        self.fireworks = alive
    
    def render(self):
        """Render scene"""
        # Ground grid
        glColor4f(0.2, 0.25, 0.3, 0.4)
        glBegin(GL_LINES)
        for i in range(-50, 51, 5):
            glVertex3f(i, 0, -50)
            glVertex3f(i, 0, 50)
            glVertex3f(-50, 0, i)
            glVertex3f(50, 0, i)
        glEnd()
        
        # Render fireworks
        for fw in self.fireworks:
            fw.render()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point"""
    if not glfw.init():
        return

    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    
    width, height = 1200, 800
    window = glfw.create_window(width, height, "3D Fireworks - Manual Transformation Matrices", None, None)
    
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    
    # OpenGL settings
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glClearColor(0.01, 0.01, 0.05, 1.0)
    
    sim = FireworkSimulation()
    keys = {}
    
    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            keys[key] = True
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_R:
                sim.auto_launch = not sim.auto_launch
                print(f"Auto-launch: {'ON' if sim.auto_launch else 'OFF'}")
            elif key == glfw.KEY_T:
                sim.cycle_type()
        elif action == glfw.RELEASE:
            keys[key] = False
    
    def mouse_callback(window, xpos, ypos):
        sim.camera.process_mouse(xpos, ypos)
    
    def scroll_callback(window, xoffset, yoffset):
        sim.camera.process_scroll(yoffset)
    
    def mouse_button_callback(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            pos = sim.camera.position + sim.camera.front * 12
            pos[1] = 0
            sim.add_firework(pos, sim.firework_types[sim.current_type])
    
    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    
    print("=" * 70)
    print("3D FIREWORKS SIMULATION - MANUAL TRANSFORMATION MATRICES")
    print("=" * 70)
    print("\nTRANSFORMATIONS IMPLEMENTED:")
    print("  ✓ Translation - Manual 4x4 matrix")
    print("  ✓ Rotation - X, Y, Z, and arbitrary axis (Rodrigues' formula)")
    print("  ✓ Scaling - Uniform and non-uniform")
    print("  ✓ Shearing - XY, XZ, YZ planes")
    print("\nCONTROLS:")
    print("  Mouse       - Look around")
    print("  W/A/S/D     - Move camera")
    print("  SPACE/SHIFT - Up/Down")
    print("  Scroll      - Zoom")
    print("  Left Click  - Launch firework")
    print("  R           - Toggle auto-launch")
    print("  T           - Cycle firework types")
    print("  ESC         - Exit")
    print("\nFIREWORK TYPES:")
    print("  1. Burst     - Rotation + Scaling")
    print("  2. Ring      - Scaling (flatten)")
    print("  3. Fountain  - Shearing + Scaling")
    print("  4. Willow    - Shearing + Rotation")
    print("=" * 70)
    print("\nStarting simulation...\n")
    
    last_time = glfw.get_time()
    
    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        dt = current_time - last_time
        last_time = current_time
        
        # Input
        if keys.get(glfw.KEY_W):
            sim.camera.position += sim.camera.front * sim.camera.speed * dt
        if keys.get(glfw.KEY_S):
            sim.camera.position -= sim.camera.front * sim.camera.speed * dt
        if keys.get(glfw.KEY_A):
            sim.camera.position -= sim.camera.right * sim.camera.speed * dt
        if keys.get(glfw.KEY_D):
            sim.camera.position += sim.camera.right * sim.camera.speed * dt
        if keys.get(glfw.KEY_SPACE):
            sim.camera.position += sim.camera.up * sim.camera.speed * dt
        if keys.get(glfw.KEY_LEFT_SHIFT):
            sim.camera.position -= sim.camera.up * sim.camera.speed * dt
        
        sim.update(dt)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(sim.camera.fov, width/height, 0.1, 500.0)
        
        # View
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye, center, up = sim.camera.get_view_matrix()
        gluLookAt(eye[0], eye[1], eye[2],
                  center[0], center[1], center[2],
                  up[0], up[1], up[2])
        
        sim.render()
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()


if __name__ == "__main__":
    main()