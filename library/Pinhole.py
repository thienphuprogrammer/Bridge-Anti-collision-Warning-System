from __future__ import division, print_function, absolute_import
import numpy as np
from typing import Tuple, Any
from scipy.optimize import fsolve


def calculate_jerk(A_s: np.ndarray, dt: float) -> np.ndarray:
    """
       Calculate the jerk J_s given the acceleration values and time interval.

       Parameters:
       A_s (np.ndarray): Array of acceleration values
       dt (float): Time interval between measurements

       Returns:
       np.ndarray: Array of jerk values
    """
    dA_dt = np.gradient(A_s, dt)
    J_s = np.abs(dA_dt)
    return J_s


def calculate_velocity(x_loc: np.ndarray, y_loc: np.ndarray, dt: float) -> np.ndarray:
    """
       Calculate the velocity V_s given the location coordinates and time interval.

       Parameters:
       x_loc (np.ndarray): Array of x location coordinates
       y_loc (np.ndarray): Array of y location coordinates
       dt (float): Time interval between measurements

       Returns:
       np.ndarray: Array of velocity values
    """
    dx_dt = np.gradient(x_loc, dt)
    dy_dt = np.gradient(y_loc, dt)
    V_s = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
    return V_s


def calculate_acceleration(V_s: np.ndarray, dt: float) -> np.ndarray:
    """
       Calculate the acceleration A_s given the velocity values and time interval.

       Parameters:
       V_s (np.ndarray): Array of velocity values
       dt (float): Time interval between measurements

       Returns:
       np.ndarray: Array of acceleration values
    """
    dV_dt = np.gradient(V_s, dt)
    A_s = np.abs(dV_dt)
    return A_s


def calculate_snap(J_s: np.ndarray, dt: float) -> np.ndarray:
    """
       Calculate the snap S_s given the jerk values and time interval.

       Parameters:
       J_s (np.ndarray): Array of jerk values
       dt (float): Time interval between measurements

       Returns:
       np.ndarray: Array of snap values
    """
    dJ_dt = np.gradient(J_s, dt)
    S_s = np.abs(dJ_dt)
    return S_s


class Pinhole:
    """
    Pinhole camera models with distortion correction.
    """
    def __init__(self, r: float, R: float, H: float, h_s_1: float,
                 h_s_2: float, d_s_1: float,
                 d_s_2: float, d_1: float, W_c: float):
        self.r = r
        self.R = R
        self.H = H
        self.h_s_1 = h_s_1
        self.h_s_2 = h_s_2
        self.d_s_1 = d_s_1
        self.d_s_2 = d_s_2
        self.d_1 = d_1
        self.W_c = W_c

        self.W_s = self.calculate_width_of_target()
        self.R_prime = self.calculate_r_prime(self.W_s)

    def calculate_width_of_target(self) -> float:
        """
        Calculate the width of the target.
        :return:
        """
        W_s = self.d_s_1 * self.W_c / self.d_1
        return W_s

    def calculate_r_prime(self, W_s: float) -> float:
        """
        Calculate R prime.
        :param W_s:
        :return:
        """
        R_prime = self.r * W_s / self.d_s_2
        return R_prime

    def calculate_height_and_length_of_target(self):
        """
        Calculate the height and length of the target.
        :return:

        """
        A = np.array([
            [self.r / self.R, self.H / (self.R * self.h_s_1), -1],
        ])

        B = np.array([self.h_s_1 * self.R, self.h_s_2 * self.R_prime])

        # A * X = B
        # shape of A = (2, 3)
        # shape of X = (3, 1)
        # shape of B = (2, 1)
        # X = (A.T * A)^-1 * A.T * B
        solution = np.linalg.inv(A.T @ A) @ A.T @ B

        return solution[0], solution[1]

    def tracking_point(self, d_s: float, d_y_loc: float) -> Tuple[float, float]:
        """
        Calculate the tracking point.

        :param d_s:
        :param d_y_loc:
        :return:
        """
        W_s = self.calculate_width_of_target()

        x_loc = (self.r * W_s) / d_s
        y_loc = (d_y_loc * self.W_c) / self.d_1

        return x_loc, y_loc


r = 0.14 #cm
R = 125 #cm
H = 20 #cm
h_s_1 = 265 #pixel
h_s_2 = 300 #pixel
d_s_1 = 291 #pixel
d_s_2 = 310 #pixel
d_1 = 400 #pixel
W_c = 50 #cm
# convert pixel to cm
inch_to_cm = 2.54
h_s_1 = h_s_1 * inch_to_cm
h_s_2 = h_s_2 * inch_to_cm
d_s_1 = d_s_1 * inch_to_cm
d_s_2 = d_s_2 * inch_to_cm
d_1 = d_1 * inch_to_cm
pinhole = Pinhole(r, R, H, h_s_1, h_s_2, d_s_1, d_s_2, d_1, W_c)
print(pinhole.calculate_height_and_length_of_target())