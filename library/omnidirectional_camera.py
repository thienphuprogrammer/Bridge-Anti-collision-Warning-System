import numpy as np


def polynomial_basis(theta: np.ndarray, degree: int) -> np.ndarray:
    """Calculates polynomial basis for the omnidirectional camera models.
    :param theta : numpy.array
        theta-angles for which the polynomial basis will be calculated for
    :param degree : int
        degree of the basis. E.g. degree = 2, basis = [1.0 theta theta^2]
    :return Returns
        numpy.array
            Polynomial basis vector/matrix. If theta = [theta1, theta2, theta3] and degree = 2, then
            the basis will be:
            [1.0, 1.0, 1.0;
            theta1, theta2, theta3;
            theta1^2, theta2^2, theta3^2]
    """

    # Minimum degree is 1
    if degree < 1:
        raise Exception("Degree has to be 1 or greater!")

    basis = np.empty((degree, theta.size), dtype=np.float32)
    basis[
        0,
    ] = np.ones((1, theta.size))

    for row in range(1, degree):
        basis[
            row,
        ] = theta

    for row in range(2, degree):
        basis[row,] *= basis[
            row - 1,
        ]

    return basis


def omnidirectional_to_perspective_lut(
    image_shape: tuple,
    principal_point: np.ndarray,
    focal_length: float,
    model_coefficients: np.ndarray,
) -> tuple:
    """
    Calculates a look-up-table (LUT) for converting images captured with an omnidirectional camera, described by
    the models coefficients, into perspective camera images (i.e. pin-hole camera). The relation between the 3D half-ray
    emanating from the single point and the corresponding pixel, observed in the image plane, is described by a
    polynomial basis and the models coefficients. The look-up-table values can be used for converting images into
    perspective camera images, for example, by using OpenCV's remap function:
    cv2.remap(image, u, v, cv2.INTER_LINEAR)

    For more information, take a look at the paper:
    "A Toolbox for Easily Calibrating Omnidirectional Cameras", D. Scaramuzza, A. Martinelli and R. Siegwart.

    :param image_shape : tuple of ints
        Shape of the image (rows, cols, channels)
    :param principal_point : (float, float)
        Principal point (i.e. optical centre of the camera) [px, py]
    :param focal_length : float
        Focal length
    :param model_coefficients :
         Coefficients of the omnidirectional lens models (https://sites.google.com/site/scarabotix/ocamcalib-toolbox)
         
    :return Returns
    (u, v)
        A tuple containing the look-up-table values for converting images into perspective camera images.
        u and v both have the same shape as image_shape (for rows and columns)
    """

    focal_length = np.abs(focal_length)

    # Create image coordinate mesh-grids. As the name implies, these are in the image coordinate system
    # with the origin at the top left corner
    u, v = np.meshgrid(
        np.arange(image_shape[1], dtype=np.float32),
        np.arange(image_shape[0], dtype=np.float32),
    )

    # Convert the coordinates into sensor coordinates (origin is at the principal point, and the
    # sensor is a focal length distance away from the lens optical centre)
    u -= principal_point[0]
    v -= principal_point[1]
    sensor_coords = np.vstack(
        (u.flatten(), v.flatten(), np.ones(u.size) * focal_length)
    )

    # Calculate the polynomial basis for the camera/lens models
    # rho is the Euclidean distance of the sensor position from the principal point
    rho = np.sqrt(np.square(sensor_coords[0, :]) + np.square(sensor_coords[1, :]))
    theta = np.arctan(
        np.divide(
            -sensor_coords[
                2,
            ],
            rho,
        )
    )
    # calculate the polynomial basis, based on the angle
    basis = polynomial_basis(theta, model_coefficients.size)

    r = np.multiply(model_coefficients.reshape((model_coefficients.size, -1)), basis)
    r = np.sum(r, axis=0)
    r /= rho

    x_result = (
        principal_point[0]
        + sensor_coords[
            0,
        ]
        * r
    )
    y_result = (
        principal_point[1]
        + sensor_coords[
            1,
        ]
        * r
    )
    x_result = x_result.reshape((image_shape[0], image_shape[1]))
    y_result = y_result.reshape((image_shape[0], image_shape[1]))

    return x_result.astype(np.float32), y_result.astype(np.float32)
