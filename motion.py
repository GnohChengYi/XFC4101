class Motion:
    def __init__(self, frame_function):
        self.frame_function = frame_function
        
    def get_frames(self, from_t, to_t):
        frames = []
        for t in range(from_t, to_t + 1):
            frames.append([t, *self.frame_function(t)])
        return frames
