import numpy as np

# Only works up to 3D (excluding time frame)
class SphericalLinearInterpolator:
    def inbetween(self, keyframes):
        if len(keyframes[0]) > 3+1:
            raise ValueError("SLERP only works up to 3D (excluding time frame)")
        interpolated_keyframes = []
        for t in range(len(keyframes)):
            keyframe = keyframes[t]
            if None not in keyframe:
                interpolated_keyframes.append(keyframe)
                continue
            interpolated_keyframe = self.interpolate(keyframes, t)
            interpolated_keyframes.append(interpolated_keyframe)
        return interpolated_keyframes
    
    def interpolate(self, keyframes, t):
        left_index, right_index = self.get_useful_keyframes(keyframes, t)
        left_x, left_y, left_z, right_x, right_y, right_z = self.get_adapted_frames(keyframes, left_index, right_index)
        
        left_r, left_theta, left_phi = self.cartesian_to_spherical(left_x, left_y, left_z)
        right_r, right_theta, right_phi = self.cartesian_to_spherical(right_x, right_y, right_z)
        
        r_interpolated = left_r + (right_r - left_r) * (t - left_index) / (right_index - left_index)
        theta_interpolated = left_theta + (right_theta - left_theta) * (t - left_index) / (right_index - left_index)
        phi_interpolated = left_phi + (right_phi - left_phi) * (t - left_index) / (right_index - left_index)
        
        interpolated_xyz = self.spherical_to_cartesian(r_interpolated, theta_interpolated, phi_interpolated)
        inbetweened_keyframe = [t, *interpolated_xyz[:len(keyframes[t])-1]]
        return inbetweened_keyframe

    # get indexes of closest non-None keyframes
    def get_useful_keyframes(self, keyframes, t):
        left_index = t
        while left_index >= 0 and None in keyframes[left_index]:
            left_index -= 1
        assert left_index >= 0, "No left index found"
        
        right_index = t
        while right_index < len(keyframes) and None in keyframes[right_index]:
            right_index += 1
        assert right_index < len(keyframes), "No right index found"
        return left_index,right_index

    # Gets left and right frames with the number of dimension topped up to 3 if necessary
    def get_adapted_frames(self, keyframes, left_index, right_index):
        left_x = keyframes[left_index][1]
        left_y = keyframes[left_index][2] if len(keyframes[0]) > 2 else 0
        left_z = keyframes[left_index][3] if len(keyframes[0]) > 3 else 0
        right_x = keyframes[right_index][1]
        right_y = keyframes[right_index][2] if len(keyframes[0]) > 2 else 0
        right_z = keyframes[right_index][3] if len(keyframes[0]) > 3 else 0
        return left_x,left_y,left_z,right_x,right_y,right_z
    
    def cartesian_to_spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-6  # avoid r=0
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi
    
    def spherical_to_cartesian(self, r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z