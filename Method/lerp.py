class LinearInterpolator:
    def inbetween(self, keyframes):
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
        
        # Perform linear interpolation for each variable (except t)
        interpolated_keyframe = [t]
        for i in range(1, len(keyframes[t])):
            left_value = keyframes[left_index][i]
            right_value = keyframes[right_index][i]
            interpolated_value = left_value + (right_value - left_value) * (t - left_index) / (right_index - left_index)
            interpolated_keyframe.append(interpolated_value)
        
        return interpolated_keyframe

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

    def __str__(self) -> str:
        return "LERP"
