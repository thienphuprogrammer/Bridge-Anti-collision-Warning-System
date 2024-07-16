from PIL import Image
import numpy as np
from PIL import ImageDraw


class Camera:
    def __init__(self, K: np.ndarray, image: Image, H: np.ndarray):
        """
        Initialize the Camera object.

        :param K: numpy.array, shape (3, 3)
                Camera intrinsic matrix
        :param image: PIL.Image
                Image to be used for the Camera object
        :param H: numpy.array, shape (3, 3)
                Homography matrix
        """
        self.K = K
        self.image = image
        self.H = H

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D using the camera intrinsic matrix and homography matrix.

        :param points: numpy.array, shape (3, n)
                3D points to be projected
        :return: numpy.array, shape (2, n)
                Projected 2D points
        """
        # Project 3D points to 2D using the camera intrinsic matrix
        points = self.K @ points

        # Project 2D points to 2D using the homography matrix
        points = self.H @ points

        # Normalize the points
        points = points / points[2, :]

        return points[:2, :]

    def draw(self, points: np.ndarray, color: str = "red") -> Image:
        """
        Draw points on the image.

        :param points: numpy.array, shape (2, n)
                2D points to be drawn
        :param color: str
                Color of the points
        :return: PIL.Image
                Image with points drawn
        """
        # Draw points on the image
        draw = ImageDraw.Draw(self.image)
        for i in range(points.shape[1]):
            draw.ellipse(
                (
                    points[0, i] - 5,
                    points[1, i] - 5,
                    points[0, i] + 5,
                    points[1, i] + 5,
                ),
                fill=color,
            )

        return self.image


# Test the Camera class
K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity matrix
image = Image.new("RGB", (100, 100))
H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
camera = Camera(K, image, H)
points = np.array([[0, 1, 2], [0, 1, 2], [1, 1, 1]])
projected_points = camera.project(points)
print(projected_points)
image_with_points = camera.draw(projected_points)
image_with_points.show()
# Expected output: Image with 3 points drawn in red color


