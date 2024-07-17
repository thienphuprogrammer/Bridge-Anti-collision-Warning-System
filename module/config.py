import numpy as np

# define extrinsic parameters

# rotate an angle of pi/4 along the standard Y axis
angles = [np.pi / 2]
order = 'z'

# translate by the given offset
offset = np.array([1, 4, 0.6])

# define intrinsic parameters
# -------------------------------

f = 0.6  # focal length
s = 0  # skew
a = 1  # aspect ratio
cx = 4  # principal point x
cy = 1  # principal point y
img_size = (100, 100)  # image size