import math
import numpy as np
import pandas as pd
import scipy.spatial.transform as st

# Only works up to 3D (excluding time frame)
class SphericalLinearInterpolator:
    def inbetween(self, keyframes):
        if len(keyframes[0]) > 3+1:
            raise ValueError("SLERP only works up to 3D (excluding time frame)")
    
        # Convert (x, y, z) to scipy Rotations
        # referred to https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html
        key_quats = []
        key_times = []
        for keyframe in keyframes:
            t = keyframe[0]
            x = keyframe[1]
            y = keyframe[2] if len(keyframe) > 2 else 0
            z = keyframe[3] if len(keyframe) > 3 else 0
            if None in (x, y, z):
                continue
            if x == y == z == 0:
                x = y = z = 1e-6    # avoid zero norm quaternions
            key_quats.append([x, y, z, 0])  # scipy uses scalar-last format
            key_times.append(t)
            
        key_rots = st.Rotation.from_quat(key_quats)
        times = [keyframe[0] for keyframe in keyframes]
        slerp = st.Slerp(key_times, key_rots)
        interp_rots = slerp(times)
        print(interp_rots.as_quat())
        
        # Convert interpolated rotations to keyframes
        inbetweened_data = []
        for i in range(len(times)):
            interp_rot = interp_rots[i]
            quat = interp_rot.as_quat()
            relevant_data = quat[:len(keyframes[0])-1]
            inbetweened_data.append([times[i], *relevant_data])
        
        return inbetweened_data
    
    # # TODO is this really slerp? lerp on polar coordinates?
    # def slerp(x1, y1, x2, y2, t):
    #     # Convert the input points to polar coordinates
    #     theta1 = math.atan2(y1, x1)
    #     theta2 = math.atan2(y2, x2)
    #     r1 = math.hypot(x1, y1)
    #     r2 = math.hypot(x2, y2)

    #     # Interpolate the polar coordinates
    #     theta = (1 - t) * theta1 + t * theta2
    #     r = (1 - t) * r1 + t * r2

    #     # Convert back to Cartesian coordinates
    #     x = r * math.cos(theta)
    #     y = r * math.sin(theta)

    #     return x, y
