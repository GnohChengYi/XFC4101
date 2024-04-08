from math import sin, cos

class Motion:
    def __init__(self, frame_function):
        self.frame_function = frame_function
        
    def get_frames(self, from_t, to_t):
        frames = []
        for t in range(from_t, to_t + 1):
            frames.append([t, *self.frame_function(t)])
        return frames
    
    def __str__(self):
        return "Motion"


class UniformlyAcceleratedMotion(Motion):    
    def __init__(self, position0, velocity0, acceleration):
        assert len(position0) == len(velocity0) == len(acceleration), "All vectors should have the same length"
        self.position0 = position0
        self.velocity0 = velocity0
        self.acceleration = acceleration
        self.position_function = lambda t: [position0[i] + velocity0[i] * t + 0.5 * acceleration[i] * t**2 for i in range(len(position0))]
        super().__init__(self.position_function)

    def __str__(self):
        return f"UniformlyAcceleratedMotion-p0={self.position0},v0={self.velocity0},a={self.acceleration}"


class EllipticalMotion(Motion):
    def __init__(self, x_radius, y_radius):
        self.x_radius = x_radius
        self.y_radius = y_radius
        self.position_function = lambda t: [x_radius * sin(t), y_radius * cos(t)]
        super().__init__(self.position_function)
    
    def __str__(self):
        return "EllipticalMotion,xr={self.x_radius},yr={self.y_radius}"


class SimpleHarmonicMotion(Motion):
    # fixed axis values are for y and z axes (or more)
    def __init__(self, amplitude, frequency, phase, fixed_axis_values):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.position_function = lambda t: [amplitude * sin(frequency * t + phase), *fixed_axis_values]
        super().__init__(self.position_function)

    def __str__(self):
        return f"SimpleHarmonicMotion-a={self.amplitude},f={self.frequency},p={self.phase}"
