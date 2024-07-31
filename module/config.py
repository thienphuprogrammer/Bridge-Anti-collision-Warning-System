import numpy as np

# define extrinsic parameters

# rotate an angle of pi/4 along the standard Y axis
angles = [np.pi]
order = 'z'

# translate by the given offset
offset = np.array([5, 4, 0.6])

# define intrinsic parameters
# -------------------------------

f = 0.6  # focal length
s = 0  # skew
a = 1  # aspect ratio
cx = 5  # principal point x
cy = 4  # principal point y
img_size = (10, 10)  # image size


# # translate by the given offset
# offset = np.array([1.5, 15.0, 2.6])
# #
# # # define intrinsic parameters
# # # -------------------------------
# #
#
# f = 2.6  # focal length
# s = 0  # skew
# a = 1   # aspect ratio
# cx = 1.5  # principal point x
# cy = 16  # principal point y
# img_size = (800, 800)  # image size