import numpy as np
from typing import Optional, Tuple
from library import utils


def backproject(
    uv: np.ndarray, K: np.ndarray, normalize: Optional[bool] = False
) -> np.ndarray:
    """Back projects 2D pixel coordinates, defined in image coordinates, into 3D direction vectors,
    defined in the camera coordinate system. The direction vectors are defined by the origin of
    the camera and the pixel position (u, v).

    If the resulting vector is not normalized, then the output vector is xi + yj + 1k.
    3D position [X, Y, Z] can be calculated, if we know the depth
    (i.e. Z-coordinate), as follows: X = x*Z, Y = y*Z and Z = Z.

    If the resulting vector is normalized, then the output vector is
    (xn*i + yn*i + zn*i) = (x*i + y*j + z*k)/|x*i + y*j + z*k|.
    3D position [X, Y, Z] can be calculated, based on the length of the ray r, as follows:
    X = xn*r, Y = yn*r, Z = zn*r.

    :param uv : np.ndarray
        Homogeneous pixel coordinates [u1, u2, u3, ...; v1, v2, v3, ...; 1, 1, 1, ...]
    :param K : np.ndarray
        3x3 camera calibration matrix [fx, s, px; 0, fy, py; 0 0 1]
    :param normalize : Optional[bool], optional
        _description_, by default False

    :return Returns
    np.ndarray
        - If `normalize=False`, 3D direction vectors [x1, x2, x3, ...; y1, y2, y3, ...; 1, 1, 1, ...]
        - If `normalize=True`, normalized 3D direction vectors [xn1, xn2, xn3, ...; yn1, yn2, yn3, ...; zn1, zn2, zn3, ...]

    :raises
    Exception
        An exception is thrown if the K matrix is not of np.ndarray type
    Exception
        An exception is thrown if the K matrix is not of shape 3x3
    Exception
        An exception is thrown if the uv matrix is not of np.ndarray type
    Exception
       An exception is thrown if the uv matrix doesn't have 3 rows (is not homogeneous)
    """

    # Verify that camera matrix K is a numpy matrix
    if not isinstance(K, np.ndarray):
        raise Exception("Camera matrix K is expected to be of numpy type")
    # Verify that camera matrix K is of size 3x3
    if K.shape != (3, 3):
        raise Exception("Camera matrix K needs to be of size 3x3")

    # Verify that uv vector is a numpy matrix
    if not isinstance(uv, np.ndarray):
        raise Exception("'uv' matrix is expected to be of numpy type")
    # Verify that uv vector is 'homogeneous'
    if uv.shape[0] != 3:
        raise Exception("'uv' matrix needs to be homogeneous: [u;v;1]'")

    result = np.matmul(np.linalg.inv(K), uv)
    # Normalize homogeneous coordinates
    result[0,] = (
        result[
            0,
        ]
        / result[
            2,
        ]
    )
    result[1,] = (
        result[
            1,
        ]
        / result[
            2,
        ]
    )
    result[2,] = (
        result[
            2,
        ]
        / result[
            2,
        ]
    )

    if normalize:
        magnitude = np.sqrt(
            np.power(
                result[
                    0,
                ],
                2,
            )
            + np.power(
                result[
                    1,
                ],
                2,
            )
            + np.power(
                result[
                    2,
                ],
                2,
            )
        )
        result[0,] = np.divide(
            result[
                0,
            ],
            magnitude,
        )
        result[1,] = np.divide(
            result[
                1,
            ],
            magnitude,
        )
        result[2,] = np.divide(
            result[
                2,
            ],
            magnitude,
        )

    return result


def forwardprojectK(
    points: np.ndarray, K: np.ndarray, image_size: Tuple, image: Optional[bool] = None
) -> Tuple[np.ndarray, ...]:
    """Projects 3D points [X Y Z]' onto an image plane of a camera defined by
    a 3x3 camera matrix K.

    Forward projects 3D points onto an image plane of a camera defined by the 3x3 camera matrix K.
    Returns those 3D points that are withing the FOV of the camera (i.e. filters out those points
    that are outside of the FOV), the corresponding uv-image coordinates, and a depth map.
    Additionally, if an image is given, RGB for each 3D point is returned.

    :param points : np.ndarray
        3D points, 3xnr_points
    :param K : np.ndarray
        Camera calibration matrix [fx, s, px; 0, fy, py; 0 0 1], 3x3
    :param image_size : Tuple
        (rows, cols)
    :param image : Optional[bool], optional
        Image used for defining colors for each projected point, by default None

    :return Returns
    Tuple[np.ndarray, ...]
        - If no input image is given, output is (3D points, uv-coordinates, depth map).
          Shapes are (3, nr_points), (3, nr_points) and (rows, cols)
        - If an input image is given, output is (3D points, uv-coordinates, RGB, depth map).
          Shapes are (3, nr_points), (3, nr_points), (3, nr_points) and (rows, cols)
    """

    # Convert the image_size into a tuple. It might already be a tuple, but let's just make sure
    image_size = (int(image_size[0]), int(image_size[1]))
    depth_map = np.ones(image_size) * np.nan

    try:
        # We expect that the points are given in the camera coordinate frame, so remove points that are
        # behind the camera, i.e. where the Z-coordinate is negative
        mask = points[2, :] <= 0.0
        points = points[:, ~mask]

        # Project points to image
        uv = np.matmul(K, points)
        # Normalize coordinates
        uv[0, :] /= uv[2, :]
        uv[1, :] /= uv[2, :]
        uv[2, :] /= uv[2, :]

        # Mask out points that don't fall withing the given image (i.e. are outside of FOV)
        mask = (
            (uv[0, :] < 0)
            | (uv[0, :] > (image_size[1] - 1))
            | (uv[1, :] < 0)
            | (uv[1, :] > (image_size[0] - 1))
        )
        points = points[:, ~mask]
        uv = uv[:, ~mask]

        # Generate a depth map
        depth_map[
            np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int)
        ] = points[2, :]

        # Handle colors, if given
        if image is None:
            return points, uv, depth_map
        else:
            RGB = image[
                np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int), :
            ]
            return points, uv, RGB, depth_map
    except Exception as e:
        print(f"forwardprojectK function raised exception: {e}")
        raise


def forwardprojectP(
    points: np.ndarray, P: np.ndarray, image_size: Tuple, image: Optional[bool] = None
) -> Tuple[np.ndarray, ...]:
    """Projects 3D points [X Y Z]' onto an image plane of a camera defined by
    a 3x4 projection matrix P.

    Forward projects 3D points onto an image plane of a camera defined by the 3x4 projection matrix P.
    Returns those 3D points that are withing the FOV of the camera (i.e. filters out those points
    that are outside of the FOV), the corresponding uv-image coordinates, and a depth map.
    Additionally, if an image is given, RGB for each 3D point is returned.

    :param points : np.ndarray
        3D points, 3xnum_points
    :param P : np.ndarray
        Camera projection matrix P = K[R | t], 3x4
    :param image_size : Tuple
        Image size (rows, cols)
    :param image : Optional[bool], optional
        Image used for defining colors for each point, by default None

    :return Returns
        Tuple[np.ndarray, ...]
            - If no image is given, output is (3D points, uv-coordinates, depth map).
              Shapes are (3, nr_points), (3, nr_points) and (rows, cols)
            - If image is given, output is (3D points, uv-coordinates, RGB, depth map).
              Shapes are (3, nr_points), (3, nr_points), (3, nr_points) and (rows, cols)
    """

    # Convert the image_size into a tuple. It might already be a tuple, but let's just make sure
    image_size = (int(image_size[0]), int(image_size[1]))
    depth_map = np.ones(image_size) * np.nan

    try:
        # Convert points into homogeneous form
        points = utils.homogenise(points)

        # Project points to image
        uv = np.matmul(P, points)

        # Filter points that fall behind the camera
        mask = uv[2, :] < 0.0
        uv = uv[:, ~mask]
        points = points[:, ~mask]

        # Normalize coordinates
        uv[0, :] /= uv[2, :]
        uv[1, :] /= uv[2, :]
        uv[2, :] /= uv[2, :]

        # Mask out points that don't fall withing the given image (i.e. are outside of FOV)
        mask = (
            (uv[0, :] < 0)
            | (uv[0, :] > (image_size[1] - 1))
            | (uv[1, :] < 0)
            | (uv[1, :] > (image_size[0] - 1))
        )
        points = points[:, ~mask]
        uv = uv[:, ~mask]

        # Generate a depth map
        depth_map[
            np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int)
        ] = points[2, :]

        # Handle colors, if given
        if image is None:
            return points[:3, :], uv, depth_map
        else:
            RGB = image[
                np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int), :
            ]
            return points[:3, :], uv, RGB, depth_map
    except Exception as e:
        print(f"forwardprojectP function raised exception: {e}")
        raise


def depthMapTo3D(
    depthMap: np.ndarray, K: np.ndarray, image: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, ...]:
    """Backprojects a depth map, defined by the Z-coordinate values, into 3D points [X Y Z]'.

    :param depthMap : np.ndarray
        Depth map, i.e. contains the Z-coordinates for each point.
    :param K : np.ndarray
        Camera calibration matrix [fx, s, px; 0, fy, py; 0 0 1], 3x3.
    :param image : Optional[np.ndarray], optional
        RGB image, if given RGB values corresponding to 3D points are returned, by default None

    :return Returns
        Tuple[np.ndarray, ...]
            - If no image is given, output is (3D_coords, uv_coord)
            - If an image is given, output is (3D_coords, uv_coords, RGB)
    """

    # Generate pixel coordinates and stack them together
    u, v = np.meshgrid(
        np.arange(depthMap.shape[1], dtype=np.float32),
        np.arange(depthMap.shape[0], dtype=np.float32),
    )
    uv_coords = np.vstack((u.flatten(), v.flatten(), np.ones(u.size)))

    # Remove nan:s
    depthMap = depthMap.flatten()
    mask = np.isnan(depthMap)
    depthMap = depthMap[~mask]
    uv_coords = uv_coords[:, ~mask]

    # Calculate normalized camera coordinates
    vector_mm = backproject(uv_coords, K)
    vector_mm[0, :] = np.multiply(vector_mm[0, :], depthMap)
    vector_mm[1, :] = np.multiply(vector_mm[1, :], depthMap)
    vector_mm[2, :] = depthMap

    if image is None:
        return vector_mm, uv_coords
    else:
        RGB = image.reshape((image[:, :, 0].size, -1))
        RGB = RGB[~mask]
        return vector_mm, image, RGB.transpose()