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


def cal_plane(points: np.ndarray) -> tuple[ndarray | ndarray, Any]:
    vec_nor = vector_normalization(points)
    d = np.dot(vec_nor, points.T[0].T)
    return vec_nor, d


def range_image(points: np.ndarray) -> tuple[ndarray | ndarray, ndarray | ndarray]:
    point_A, point_B, point_C, point_D = points.T
    line_AB = ray_tracing(np.array([point_A, point_B]))
    line_CD = ray_tracing(np.array([point_C, point_D]))
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
z = [14, 14, 1, 1]
y = [1, 7, 7, 1]
x = [0, 0, 0, 0]
rand_points = np.vstack((x, y, z))

project_points = camera.project_3d_point_to_2d(rand_points, is_homogeneous=False)
line_1, line_2 = range_image(project_points)

new_point = [0, 3.23, 1.44]
new_line = ray_tracing(np.array([offset, new_point]))

plane_h = cal_plane(rand_points)
intersection = intersection_between_line_and_plane(new_line, plane_h)
# cal line original with the plane and the intersection point
# point_or = np.array([0, 3.23, 0.87])
# point_or = np.vstack((point_or, intersection))
# line_or = line_original_with_plane(intersection, plane_h)
# print(line_or)
# intersection_or = intersection_between_line_and_plane(line_or, plane_h)

# Test the function
x_ = [1, 0, 1, 0]
y_ = [4.5, 4.5, 4.5, 4.5]
z_ = [9, 8, 8, 7]
rand_points_ = np.vstack((x_, y_, z_))

project_points_ = camera.project_3d_point_to_2d(rand_points_, is_homogeneous=False)
line_1_, line_2_ = range_image(project_points_)
r = f
R = abs(offset[2] -z_[2])
H = offset[0]
print(project_points_)
h_s_1 = abs(project_points_[0][0] - project_points_[0][1])
h_s_2 = abs(project_points_[0][2] - project_points_[0][3])
d_s_1 = abs(project_points_[1][0] - project_points_[1][1])
d_s_2 = abs(project_points_[1][2] - project_points_[1][3])
d_1 = 0.27
W_c = abs(y[1] - y[0])
pinhole = Pinhole(r, R, H, h_s_1, h_s_2, d_s_1, d_s_2, d_1, W_c)
print(pinhole.calculate_height_and_length_of_target())