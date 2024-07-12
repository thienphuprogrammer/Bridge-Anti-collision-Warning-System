import numpy as np


def rotation(rot_z: float, rot_y: float, rot_x: float) -> np.ndarray:
    """Calculate a 3D rotation, defined by a rotation along the z-, y- and z-axis
    :param rot_z : float
        Rotation with respect the z-axis, in radians
    :param rot_y : float
        Rotation with respect the y-axis, in radians
    :param rot_x : float
        Rotation with respect the x-axis, in radians
    :return Returns
        numpy.ndarray, shape (3, 3)
            A 3x3 rotation matrix
    """
    rz = np.array(
        [
            [np.cos(rot_z), -np.sin(rot_z), 0.0],
            [np.sin(rot_z), np.cos(rot_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [np.cos(rot_y), 0.0, np.sin(rot_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(rot_y), 0.0, np.cos(rot_y)],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rot_x), -np.sin(rot_x)],
            [0.0, np.sin(rot_x), np.cos(rot_x)],
        ]
    )

    return rz @ ry @ rx


def homogenise(points: np.ndarray) -> np.ndarray:
    """Converts non-homogeneous 2D or 3D coordinates into homogeneous coordinates.
    For example
    [x1 x2 x3]    [x1 x2 x3]
    [y1 y2 y3] -> [y1 y2 y3]
                  [1  1  1]

    :param points : np.ndarray
        Points to be homogenized, 2xnum_points

    :return Returns
        np.ndarray
            Homogenized points, 3xnum_points
    """

    return np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)


def concatenateRt(R: np.array, t: np.ndarray) -> np.ndarray:
    """Concatenates a rotation matrix R and a translation vector t into a transformation matrix T.

    The outcome is as follows:
    [ R      | t]
    [[0 0 0] | 1]

    :param R : np.array
        Rotation matrix, 3x3.
    :param t : np.ndarray
        Translation vector, 3x1.

    :return Returns
        np.ndarray
            Transformation matrix, 3x4.
    """

    return np.concatenate((np.hstack((R, t)), np.array([[0, 0, 0, 1]])), axis=0)


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply transformation T to 3D points.

    For example, if the transformation T is defined as follows:
    [ R      | t]
    [[0 0 0] | 1]

    , where R is a 3x3 rotation matrix and t is a 3x1 translation vector,
    then the results is as follows:

    [ R      | t] * [P]
    [[0 0 0] | 1]   [1]
    , where P is a 3xn vector of n points. 3D points P are homogenized automatically.

    :param T : np.ndarray
        Transformation matrix, 4x4
    :param points : np.ndarray
        3D points to be transformed, 3xnum_points

    :return Returns
    np.ndarray
        Transformed points, 3xnum_points

    :raises
    Exception
        If the transformation fails an exception is thrown
    """

    try:
        points = homogenise(points)
        points = np.matmul(T, points)
        return points[:3, :]
    except Exception as e:
        print(f"transform-function raised an exception: {e}")
        raise