import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from module.config import *
from module.utils import *
from module.Pinhole import *


def visualize_2d_projection(points: np.ndarray,
                            color: str = "orange",
                            title: str = "projection of points in the image") -> None:
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)

    for i in range(points.shape[1]):
        ax.scatter(*points.T[i], color=color)
    ax.set_title(title)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.show()


def ray_tracing(points: np.ndarray) -> np.ndarray:
    # compute the ray tracing of the 3d points
    # points: (3, n)
    point_1, point_2 = points
    vec_u = point_2 - point_1

    # create a line
    line = np.array([point_1, vec_u])
    return line


def vector_normalization(points: np.ndarray) -> np.ndarray:
    # normalize the vector
    # points: (3, 3)
    points = points.T
    vec_u = points[1] - points[0]
    vec_v = points[2] - points[0]
    # (dimension must be 2 or 3)
    vec_u = vec_u.reshape(1, -1)
    vec_v = vec_v.reshape(1, -1)
    vec_normal = np.cross(vec_u, vec_v)
    return vec_normal / np.linalg.norm(vec_normal)


def plane_location(points: np.ndarray) -> tuple[ndarray | ndarray, Any]:
    vec_nor = vector_normalization(points)
    d = np.dot(vec_nor, points.T[0].T)
    return vec_nor, d


def range_image(points: np.ndarray) -> tuple[ndarray | ndarray, ndarray | ndarray]:
    point_A, point_B, point_C, point_D = points.T
    line_AB = ray_tracing(np.array([point_A, point_D]))
    line_CD = ray_tracing(np.array([point_C, point_B]))
    return line_AB, line_CD


def intersection_between_line_and_plane(line: np.ndarray, plane) -> np.ndarray:
    # compute the intersection between the line and the plane
    # line: (2, 3)
    # plane: (3, 4)
    point_1, vec_u = line
    vec_n, d = plane

    t = (d - np.dot(vec_n, point_1)) / np.dot(vec_n, vec_u)
    point = point_1 + t * vec_u
    return point


def project_points_to_line(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    # project the points to the line
    # points: (3, n)
    # line: (2, 3)
    point_1, vec_u = line
    vec_u = vec_u / np.linalg.norm(vec_u)
    vec_u = vec_u.reshape(1, -1)
    points = points.T
    points = points - point_1
    points = points - np.dot(points, vec_u.T) * vec_u
    return points.T


def line_original_with_plane(point: np.ndarray, plane) -> np.ndarray:
    # compute the line original with the plane
    # point: (3,)
    # plane: (3, 4)
    vec_n, d = plane
    return np.array([point, vec_n])


class Camera:
    def __init__(self):
        self.R = create_rotation_transformation_matrix(angles, order)
        self.R_ = np.identity(4)
        self.R_[:3, :3] = self.R

        # create translation transformation matrix
        self.T_ = create_translation_matrix(offset)
        self.E = np.linalg.inv(self.T_ @ self.R_)
        self.E = self.E[:-1, :]

        # create intrinsic matrix
        self.K = compute_intrinsic_parameter_matrix(f, s, a, cx, cy)

    def project_3d_point_to_2d(self, points: np.ndarray, is_homogeneous: bool = False) -> np.ndarray:
        rand_points_camera = compute_coordniates_wrt_camera(points, self.E, is_homogeneous=is_homogeneous)
        projections = compute_image_projection(rand_points_camera, self.K)
        return projections


def check_points_in_range(point: np.ndarray, lines: np.ndarray) -> bool:
    # check if the points are in the range
    line_1, line_2 = lines

    project_point_1 = project_points_to_line(point, line_1)
    project_point_2 = project_points_to_line(point, line_2)

    if point[1] < project_point_1[1] or point[1] > project_point_2[1]:
        return False
    return True


# create a camera object
camera = Camera()
n_points = 4
z = [14, 14, 2, 2]
y = [-4, 11, 11, -4]
x = [0, 0, 0, 0]
rand_points = np.vstack((x, y, z))

project_points = camera.project_3d_point_to_2d(rand_points, is_homogeneous=False)
print(project_points[:, 3])
line_1, line_2 = range_image(project_points)

# create an image grid
xx, yy, Z = create_image_grid(f, img_size)
# convert the image grid to homogeneous coordinates
pt_h = convert_grid_to_homogeneous(xx, yy, Z, img_size)
# transform the homogeneous coordinates
pt_h_transformed = camera.T_ @ camera.R_ @ pt_h
# convert the transformed homogeneous coordinates back to the image grid
xxt, yyt, Zt = convert_homogeneous_to_grid(pt_h_transformed, img_size)
rec = 20
# define axis and figure
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111,projection='3d')  # 111 is nrows, ncols, index

# set limits
ax.set(xlim=(0, rec), ylim=(0, rec), zlim=(0, rec))

# plot the camera in the world
ax = pr.plot_basis(ax, camera.R, offset)
ax.plot_surface(xxt, yyt, Zt, alpha=0.25)

# plot the generated random points
c = 0
for i in range(n_points):
    point = rand_points[:, c]
    ax.scatter(*point, color="orange")
    ax.plot(*make_line(offset, point), color="purple", alpha=0.25)
    c += 1

ax.set_title("The Setup")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()


#
# new_point = [5.56, 4, 0]
# new_line = ray_tracing(np.array([offset, new_point]))
#
# plane_h = plane_location(rand_points)
# # intersection = intersection_between_line_and_plane(new_line, plane_h)
#
# # Test the function
# point_1 = np.array([5.19, 4, 0])
# point_2 = np.array([5.41, 4, 0])
# point_3 = np.array([5.56, 4, 0])
# point_4 = np.array([5.24, 4, 0])
#
# point_5 = np.array([5.41, 4.16, 0])
# point_6 = np.array([5.41, 3.84, 0])
# point_7 = np.array([5.56, 4.22, 0])
# point_8 = np.array([5.56, 3.78, 0])
# # points = np.vstack((point_1, point_2, point_3, point_4))
# # new_line_1 = ray_tracing(np.array([offset, point_1]))
# # new_line_2 = ray_tracing(np.array([offset, point_2]))
# new_line_3 = ray_tracing(np.array([offset, point_3]))
# # new_line_4 = ray_tracing(np.array([offset, point_4]))
# # intersection_1 = intersection_between_line_and_plane(new_line_1, plane_h)
# # intersection_2 = intersection_between_line_and_plane(new_line_2, plane_h)
# intersection_3 = intersection_between_line_and_plane(new_line_3, plane_h)
# # intersection_4 = intersection_between_line_and_plane(new_line_4, plane_h)
#
# r = f
# R = abs(offset[2] - intersection_3[2])
# H = offset[0]
# h_s_1 = abs(point_1[0] - point_3[0])
# h_s_2 = abs(point_2[0] - point_4[0])
# d_s_1 = abs(point_5[1] - point_6[1])
# d_s_2 = abs(point_7[1] - point_8[1])
#
# # calculate the width of the target
# point_old_1, vec_u = line_1
# t_1 = (point_old_1[1] - point_5[1]) / vec_u[1]
# point_new_1 = point_old_1 + t_1 * vec_u
#
# point_old_2, vec_u = line_2
# t_2 = (point_old_2[1] - point_6[1]) / vec_u[1]
# point_new_2 = point_old_2 + t_2 * vec_u
# d_1 = abs(point_new_1[1] - point_new_2[1])
# W_c = abs(y[1] - y[0])
#
# print(f"r: {r}")
# print(f"R: {R}")
# print(f"H: {H}")
# print(f"h_s_1: {h_s_1}")
# print(f"h_s_2: {h_s_2}")
# print(f"d_s_1: {d_s_1}")
# print(f"d_s_2: {d_s_2}")
# print(f"line_1: {line_1}")
# print(f"line_2: {line_2}")
# print(f"point_old_1: {point_old_1}")
# print(f"point_old_2: {point_old_2}")
# print(f"new_point_1: {point_new_2}")
# print(f"new_point_2: {point_new_1}")
# print(f"d_1: {d_1}")
# print(f"W_c: {W_c}")
#
#
# pinhole = Pinhole(r, R, H, h_s_1, h_s_2, d_s_1, d_s_2, d_1, W_c)
# print(pinhole.calculate_height_and_length_of_target())