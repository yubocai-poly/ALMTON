"""
Test function definitions (objective, gradient, Hessian, and third-order derivative).

Adapted from the codebase accompanying:
    Olha Silina and Jeffrey Zhang, "An Unregularized Third Order Newton Method," 2023. (https://arxiv.org/abs/2209.10051)
    https://github.com/jeffreyzhang92/Third_Order_Newton/blob/main/Newton_Functions.py
"""


# Supported Functions: Beale, Bohachevsky, Goldstein, Himmelblau, McCormick, Styblinski

import numpy as np
import math


def init_func(func_name):

    if func_name == "Beale":

        def fX(X, Y):
            return (
                (1.5 - X + X * Y) ** 2
                + (2.25 - X + X * Y**2) ** 2
                + (2.625 - X + X * Y**3) ** 2
            )

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return (
                (1.5 - x + x * y) ** 2
                + (2.25 - x + x * y**2) ** 2
                + (2.625 - x + x * y**3) ** 2
            )

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return np.array(
                [
                    [
                        2 * x * (y**6 + y**4 - 2 * y**3 - y**2 - 2 * y + 3)
                        + 5.25 * y**3
                        + 4.5 * y**2
                        + 3 * y
                        - 12.75
                    ],
                    [
                        6
                        * x
                        * (
                            x * (y**5 + (2 / 3) * y**3 - y**2 - (1 / 3) * y - 1 / 3)
                            + 2.625 * y**2
                            + 1.5 * y
                            + 0.5
                        )
                    ],
                ]
            )

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2))
            ret[0, 0] = 2 * y**6 + 2 * y**4 - 4 * y**3 - 2 * y**2 - 4 * y + 6
            ret[0, 1] = ret[1, 0] = (
                12 * x * y**5
                + 8 * x * y**3
                - 12 * x * y**2
                - 4 * x * y
                - 4 * x
                + 15.75 * y**2
                + 9 * y
                + 3
            )
            ret[1, 1] = (
                30 * x**2 * y**4
                + 12 * x**2 * y**2
                - 12 * x**2 * y
                - 2 * x**2
                + 31.5 * x * y
                + 9 * x
            )
            return ret

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2, 2))
            ret[0, 0, 1] = ret[0, 1, 0] = ret[1, 0, 0] = (
                12 * y**5 + 8 * y**3 - 12 * y**2 - 4 * y - 4
            )
            ret[1, 1, 0] = ret[1, 0, 1] = ret[0, 1, 1] = (
                60 * x * y**4 + 24 * x * y**2 - 24 * x * y - 4 * x + 31.5 * y + 9
            )
            ret[1, 1, 1] = 120 * x**2 * y**3 + 24 * x**2 * y - 12 * x**2 + 31.5 * x
            return ret

    elif func_name == "Bohachevsky":

        def fX(X, Y):
            return (
                X**2
                + 2 * Y**2
                - 0.3 * np.cos(3 * np.pi * X)
                - 0.4 * np.cos(4 * np.pi * Y)
                + 0.7
            )

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return (
                x**2
                + 2 * y**2
                - 0.3 * np.cos(3 * np.pi * x)
                - 0.4 * np.cos(4 * np.pi * y)
                + 0.7
            )

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return np.array(
                [
                    [2 * x + 0.9 * np.pi * np.sin(3 * np.pi * x)],
                    [4 * y + 1.6 * np.pi * np.sin(4 * np.pi * y)],
                ]
            )

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2))
            ret[0, 0] = 2 + 2.7 * np.pi**2 * np.cos(3 * np.pi * x)
            ret[0, 1] = ret[1, 0] = 0
            ret[1, 1] = 4 + 6.4 * np.pi**2 * np.cos(4 * np.pi * y)
            return ret

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2, 2))
            ret[0, 0, 0] = -8.1 * np.pi**3 * np.sin(3 * np.pi * x)
            ret[0, 0, 1] = ret[0, 1, 0] = ret[1, 0, 0] = 0
            ret[1, 1, 0] = ret[1, 0, 1] = ret[0, 1, 1] = 0
            ret[1, 1, 1] = -25.6 * np.pi**3 * np.sin(4 * np.pi * y)
            return ret

    elif func_name == "Goldstein":

        def fX(X, Y):
            return (
                1
                + (X + Y + 1) ** 2
                * (19 - 4 * X + 3 * X**2 - 14 * Y + 6 * X * Y + 3 * Y**2)
            ) * (
                30
                + (2 * X - 3 * Y) ** 2
                * (18 - 32 * X + 12 * X**2 + 48 * Y - 36 * X * Y + 27 * Y**2)
            )

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return (
                1
                + (x + y + 1) ** 2
                * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
            ) * (
                30
                + (2 * x - 3 * y) ** 2
                * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
            )

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return np.array(
                [
                    [
                        24
                        * (
                            8 * x**3
                            - 4 * x**2 * (9 * y + 4)
                            + 6 * x * (9 * y**2 + 8 * y + 1)
                            - 9 * y * (3 * y**2 + 4 * y + 1)
                        )
                        * (
                            (3 * x**2 + 2 * x * (3 * y - 7) + 3 * y**2 - 14 * y + 19)
                            * (x + y + 1) ** 2
                            + 1
                        )
                        + 12
                        * (
                            x**3
                            + x**2 * (3 * y - 2)
                            + x * (3 * y**2 - 4 * y - 1)
                            + y**3
                            - 2 * y**2
                            - y
                            + 2
                        )
                        * (
                            (
                                12 * x**2
                                - 4 * x * (9 * y + 8)
                                + 3 * (9 * y**2 + 16 * y + 6)
                            )
                            * (2 * x - 3 * y) ** 2
                            + 30
                        )
                    ],
                    [
                        12
                        * (
                            x**3
                            + x**2 * (3 * y - 2)
                            + x * (3 * y**2 - 4 * y - 1)
                            + y**3
                            - 2 * y**2
                            - y
                            + 2
                        )
                        * (
                            (
                                12 * x**2
                                - 4 * x * (9 * y + 8)
                                + 3 * (9 * y**2 + 16 * y + 6)
                            )
                            * (2 * x - 3 * y) ** 2
                            + 30
                        )
                        - 36
                        * (
                            8 * x**3
                            - 4 * x**2 * (9 * y + 4)
                            + 6 * x * (9 * y**2 + 8 * y + 1)
                            - 9 * y * (3 * y**2 + 4 * y + 1)
                        )
                        * (
                            (3 * x**2 + 2 * x * (3 * y - 7) + 3 * y**2 - 14 * y + 19)
                            * (x + y + 1) ** 2
                            + 1
                        )
                    ],
                ]
            )

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2))
            ret[0, 0] = 12 * (
                672 * x**6
                - 336 * x**5 * (3 * y + 8)
                - 20 * x**4 * (81 * y**2 - 168 * y - 119)
                + 40 * x**3 * (51 * y**3 + 108 * y**2 - 7 * y + 56)
                + 3 * x**2 * (435 * y**4 - 1360 * y**3 - 1790 * y**2 - 2560 * y - 818)
                - 2
                * x
                * (459 * y**5 + 870 * y**4 - 310 * y**3 - 2460 * y**2 - 1446 * y + 268)
                - 3
                * (
                    81 * y**6
                    - 204 * y**5
                    - 485 * y**4
                    - 280 * y**3
                    - 432 * y**2
                    - 408 * y
                    - 70
                )
            )
            ret[0, 1] = ret[1, 0] = -12 * (
                168 * x**6
                + 24 * x**5 * (27 * y - 28)
                - 10 * x**4 * (153 * y * 2 + 216 * y - 7)
                - 20 * x**3 * (87 * y**3 - 204 * y**2 - 179 * y - 128)
                + 3 * x**2 * (765 * y**4 + 1160 * y**3 - 310 * y**2 - 1640 * y - 482)
                + 6
                * x
                * (243 * y**5 - 510 * y**4 - 970 * y**3 - 420 * y**2 - 432 * y - 204)
                - 567 * y**6
                - 972 * y**5
                + 495 * y**4
                + 3960 * y**3
                + 5904 * y**2
                + 3216 * y
                + 390
            )
            ret[1, 1] = -12 * (
                108 * x**6
                - 36 * x**5 * (17 * y + 12)
                + x**4 * (-1305 * y**2 + 2040 * y + 895)
                + 20 * x**3 * (153 * y**3 + 174 * y**2 - 31 * y - 82)
                + 9 * x**2 * (405 * y**4 - 680 * y**3 - 970 * y**2 - 280 * y - 144)
                - 6
                * x
                * (567 * y**5 + 810 * y**4 - 330 * y**3 - 1980 * y**2 - 1968 * y - 536)
                - 6
                * (
                    567 * y**6
                    - 378 * y**5
                    - 1845 * y**4
                    + 540 * y**3
                    + 2391 * y**2
                    + 1024 * y
                    + 85
                )
            )
            return ret

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2, 2))
            ret[0, 0, 0] = 24 * (
                2016 * x**5
                - 840 * x**4 * (3 * y + 8)
                - 40 * x**3 * (81 * y**2 - 168 * y - 119)
                + 60 * x**2 * (51 * y**3 + 108 * y**2 - 7 * y + 56)
                + 3 * x * (435 * y**4 - 1360 * y**3 - 1790 * y**2 - 2560 * y - 818)
                - 459 * y**5
                - 870 * y**4
                + 310 * y**3
                + 2460 * y**2
                + 1446 * y
                - 268
            )
            ret[0, 0, 1] = ret[0, 1, 0] = ret[1, 0, 0] = -24 * (
                504 * x**5
                + 60 * x**4 * (27 * y - 28)
                - 20 * x**3 * (153 * y**2 + 216 * y - 7)
                - 30 * x**2 * (87 * y**3 - 204 * y**2 - 179 * y - 128)
                + 3 * x * (765 * y**4 + 1160 * y**3 - 310 * y**2 - 1640 * y - 482)
                + 3
                * (243 * y**5 - 510 * y**4 - 970 * y**3 - 420 * y**2 - 432 * y - 204)
            )
            ret[1, 1, 0] = ret[1, 0, 1] = ret[0, 1, 1] = -24 * (
                324 * x**5
                - 90 * x**4 * (17 * y + 12)
                - 10 * x**3 * (261 * y**2 - 408 * y - 179)
                + 30 * x**2 * (153 * y**3 + 174 * y**2 - 31 * y - 82)
                + 9 * x * (405 * y**4 - 680 * y**3 - 970 * y**2 - 280 * y - 144)
                - 3
                * (567 * y**5 + 810 * y**4 - 330 * y**3 - 1980 * y**2 - 1968 * y - 536)
            )
            ret[1, 1, 1] = 24 * (
                306 * x**5
                + 15 * x**4 * (87 * y - 68)
                - 10 * x**3 * (459 * y**2 + 348 * y - 31)
                - 90 * x**2 * (81 * y**3 - 102 * y**2 - 97 * y - 14)
                + 9 * x * (945 * y**4 + 1080 * y**3 - 330 * y**2 - 1320 * y - 656)
                + 6
                * (1701 * y**5 - 945 * y**4 - 3690 * y**3 + 810 * y**2 + 2391 * y + 512)
            )
            return ret

    elif func_name == "Himmelblau":

        def fX(X, Y):
            return (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2 + 0.001

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return np.array(
                [
                    [4 * x**3 + 4 * x * y - 42 * x + 2 * y**2 - 14],
                    [2 * x**2 + 4 * x * y + 4 * y**3 - 26 * y - 22],
                ]
            )

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2))
            ret[0, 0] = 12 * x**2 + 4 * y - 42
            ret[0, 1] = ret[1, 0] = 4 * x + 4 * y
            ret[1, 1] = 4 * x + 12 * y**2 - 26
            return ret

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2, 2))
            ret[0, 0, 0] = 24 * x
            ret[0, 0, 1] = ret[0, 1, 0] = ret[1, 0, 0] = 4
            ret[1, 1, 0] = ret[1, 0, 1] = ret[0, 1, 1] = 4
            ret[1, 1, 1] = 24 * y
            return ret

    elif func_name == "McCormick":

        def fX(X, Y):
            return np.sin(X + Y) + (X - Y) ** 2 - 1.5 * X + 2.5 * Y + 3

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return np.array(
                [
                    [math.cos(x + y) + 2 * (x - y) - 1.5],
                    [math.cos(x + y) - 2 * (x - y) + 2.5],
                ]
            )

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2))
            ret[0, 0] = -math.sin(x + y) + 2
            ret[0, 1] = ret[1, 0] = -math.sin(x + y) - 2
            ret[1, 1] = -math.sin(x + y) + 2
            return ret

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2, 2))
            ret[0, 0, 0] = -math.cos(x + y)
            ret[0, 0, 1] = ret[0, 1, 0] = ret[1, 0, 0] = -math.cos(x + y)
            ret[1, 1, 0] = ret[1, 0, 1] = ret[0, 1, 1] = -math.cos(x + y)
            ret[1, 1, 1] = -math.cos(x + y)
            return ret

    elif func_name == "Styblinski":

        def fX(X, Y):
            return (
                0.5 * X**4 - 8 * X**2 + 2.5 * X + 0.5 * Y**4 - 8 * Y**2 + 2.5 * Y + 80
            )

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return 0.5 * x**4 - 8 * x**2 + 2.5 * x + 0.5 * y**4 - 8 * y**2 + 2.5 * y

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            return np.array([[2 * x**3 - 16 * x + 2.5], [2 * y**3 - 16 * y + 2.5]])

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2))
            ret[0, 0] = 6 * x**2 - 16
            ret[0, 1] = ret[1, 0] = 0
            ret[1, 1] = 6 * y**2 - 16
            return ret

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            ret = np.zeros((2, 2, 2))
            ret[0, 0, 0] = 12 * x
            ret[0, 0, 1] = ret[0, 1, 0] = ret[1, 0, 0] = 0
            ret[1, 1, 0] = ret[1, 0, 1] = ret[0, 1, 1] = 0
            ret[1, 1, 1] = 12 * y
            return ret

    elif func_name == "Slalom":
        # Slalom function from Cartis AR3 paper - complex function with third-order structure
        # This function is designed to "trap" second-order methods (AR2/Newton)
        # Only third-order methods can "see" the correct descent path
        
        # Helper functions
        def sigmoid_fun(x):
            expminusx = np.exp(-x)
            fun = 1.0 / (1.0 + expminusx)
            der1f = expminusx / (1.0 + expminusx) ** 2
            der2f = (-expminusx * (1.0 - expminusx)) / (1.0 + expminusx) ** 3
            der3f = (expminusx - 4 * expminusx**2 + expminusx**3) / (1.0 + expminusx) ** 4
            return fun, der1f, der2f, der3f

        def scaling_helper(x):
            poly = np.array([-20, 70, -84, 35, 0, 0, 0, 0])
            fun = np.polyval(poly, x)
            poly_der1 = np.polyder(poly)
            der1f = np.polyval(poly_der1, x)
            poly_der2 = np.polyder(poly_der1)
            der2f = np.polyval(poly_der2, x)
            poly_der3 = np.polyder(poly_der2)
            der3f = np.polyval(poly_der3, x)
            return fun, der1f, der2f, der3f

        def wiggly_helper(x):
            poly = np.array([6, -15, 10, 0, 0, 0])
            fun = np.polyval(poly, x)
            poly_der1 = np.polyder(poly)
            der1f = np.polyval(poly_der1, x)
            poly_der2 = np.polyder(poly_der1)
            der2f = np.polyval(poly_der2, x)
            poly_der3 = np.polyder(poly_der2)
            der3f = np.polyval(poly_der3, x)
            return fun, der1f, der2f, der3f

        def wiggly(x):
            x_mod = np.mod(x + 0.5, 1.0)
            fun_h, der1h, der2h, der3h = wiggly_helper(x_mod)
            fun = fun_h + np.floor(x + 0.5) - 0.5
            return fun, der1h, der2h, der3h

        def scaling(x, y):
            if x <= -0.5:
                xtilde = 2 * x + 2
                h, der1h, der2h, der3h = scaling_helper(xtilde)
                fun = -h * (1 - y) + 1
                der1f = np.array([-der1h * 2 * (1 - y), h])
                der2f = np.array([[-der2h * 4 * (1 - y), der1h * 2], [der1h * 2, 0]])
                der3f = np.zeros((2, 2, 2))
                der3f[0, 0, 0] = -der3h * 8 * (1 - y)
                der3f[0, 0, 1] = der3f[0, 1, 0] = der3f[1, 0, 0] = der2h * 4
                return fun, der1f, der2f, der3f
            elif x < 0.5:
                fun = y
                der1f = np.array([0.0, 1.0])
                der2f = np.zeros((2, 2))
                der3f = np.zeros((2, 2, 2))
                return fun, der1f, der2f, der3f
            else:
                xtilde = 2 * x - 1
                h, der1h, der2h, der3h = scaling_helper(xtilde)
                fun = h * (1 - y) + y
                der1f = np.array([der1h * 2 * (1 - y), -h + 1])
                der2f = np.array([[der2h * 4 * (1 - y), -der1h * 2], [-der1h * 2, 0]])
                der3f = np.zeros((2, 2, 2))
                der3f[0, 0, 0] = der3h * 8 * (1 - y)
                der3f[0, 0, 1] = der3f[0, 1, 0] = der3f[1, 0, 0] = -der2h * 4
                return fun, der1f, der2f, der3f

        def step(x, y):
            sig, der1sig, der2sig, der3sig = sigmoid_fun(y)
            s, der1s, der2s, der3s = scaling(x, 2 * sig)
            w, der1w, der2w, der3w = wiggly(x)
            fun = w * s
            der1f = np.zeros(2)
            der1f[0] = der1w * s + w * der1s[0]
            der1f[1] = w * der1s[1] * 2 * der1sig
            der2f = np.zeros((2, 2))
            der2f[0, 0] = der2w * s + 2 * der1w * der1s[0] + w * der2s[0, 0]
            der2f[0, 1] = (der1w * der1s[1] + w * der2s[0, 1]) * 2 * der1sig
            der2f[1, 0] = der2f[0, 1]
            der2f[1, 1] = w * (der2s[1, 1] * 4 * der1sig**2 + der1s[1] * 2 * der2sig)
            der3f = np.zeros((2, 2, 2))
            der3f[0, 0, 0] = der3w * s + 3 * der2w * der1s[0] + 3 * der1w * der2s[0, 0] + w * der3s[0, 0, 0]
            der3f[0, 0, 1] = (der2w * der1s[1] + 2 * der1w * der2s[0, 1] + w * der3s[0, 0, 1]) * 2 * der1sig
            der3f[0, 1, 0] = der3f[0, 0, 1]
            der3f[0, 1, 1] = der1w * (der2s[1, 1] * 4 * der1sig**2 + der1s[1] * 2 * der2sig) + w * (der3s[0, 1, 1] * 4 * der1sig**2 + der2s[0, 1] * 2 * der2sig)
            der3f[1, 0, 0] = der3f[0, 0, 1]
            der3f[1, 0, 1] = der3f[0, 1, 1]
            der3f[1, 1, 0] = der3f[0, 1, 1]
            der3f[1, 1, 1] = w * (der3s[1, 1, 1] * 8 * der1sig**3 + 3 * der2s[1, 1] * 4 * der1sig * der2sig + der1s[1] * 2 * der3sig)
            return fun, der1f, der2f, der3f

        def fX(X, Y):
            return np.zeros_like(X)

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            if y <= 0:
                fun, _, _, _ = step(np.mod(x + 1, 2) - 1, y)
                fun = fun + 2 * np.floor((x + 1) / 2)
            else:
                fun, _, _, _ = step(np.mod(x, 2) - 1, -y)
                fun = fun + 2 * np.floor(x / 2) + 1
            fun = fun + 3e-4 * x
            return fun

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            if y <= 0:
                _, der1f, _, _ = step(np.mod(x + 1, 2) - 1, y)
            else:
                _, der1f, _, _ = step(np.mod(x, 2) - 1, -y)
                der1f[1] = -der1f[1]
            der1f[0] = der1f[0] + 3e-4
            return der1f.reshape(-1, 1)

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            if y <= 0:
                _, _, der2f, _ = step(np.mod(x + 1, 2) - 1, y)
            else:
                _, _, der2f, _ = step(np.mod(x, 2) - 1, -y)
                der2f[0, 1] = -der2f[0, 1]
                der2f[1, 0] = -der2f[1, 0]
            return der2f

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            if y <= 0:
                _, _, _, der3f = step(np.mod(x + 1, 2) - 1, y)
            else:
                _, _, _, der3f = step(np.mod(x, 2) - 1, -y)
                der3f[0, 0, 1] = -der3f[0, 0, 1]
                der3f[0, 1, 0] = -der3f[0, 1, 0]
                der3f[1, 0, 0] = -der3f[1, 0, 0]
                der3f[1, 1, 1] = -der3f[1, 1, 1]
            return der3f

    elif func_name == "HairpinTurn":
        # Hairpin Turn function from Cartis AR3 paper
        # Similar structure to Slalom but with barrier terms
        
        # Helper functions (same as Slalom)
        def sigmoid_fun(x):
            expminusx = np.exp(-x)
            fun = 1.0 / (1.0 + expminusx)
            der1f = expminusx / (1.0 + expminusx) ** 2
            der2f = (-expminusx * (1.0 - expminusx)) / (1.0 + expminusx) ** 3
            der3f = (expminusx - 4 * expminusx**2 + expminusx**3) / (1.0 + expminusx) ** 4
            return fun, der1f, der2f, der3f

        def scaling_helper(x):
            poly = np.array([-20, 70, -84, 35, 0, 0, 0, 0])
            fun = np.polyval(poly, x)
            poly_der1 = np.polyder(poly)
            der1f = np.polyval(poly_der1, x)
            poly_der2 = np.polyder(poly_der1)
            der2f = np.polyval(poly_der2, x)
            poly_der3 = np.polyder(poly_der2)
            der3f = np.polyval(poly_der3, x)
            return fun, der1f, der2f, der3f

        def wiggly_helper(x):
            poly = np.array([6, -15, 10, 0, 0, 0])
            fun = np.polyval(poly, x)
            poly_der1 = np.polyder(poly)
            der1f = np.polyval(poly_der1, x)
            poly_der2 = np.polyder(poly_der1)
            der2f = np.polyval(poly_der2, x)
            poly_der3 = np.polyder(poly_der2)
            der3f = np.polyval(poly_der3, x)
            return fun, der1f, der2f, der3f

        def wiggly(x):
            x_mod = np.mod(x + 0.5, 1.0)
            fun_h, der1h, der2h, der3h = wiggly_helper(x_mod)
            fun = fun_h + np.floor(x + 0.5) - 0.5
            return fun, der1h, der2h, der3h

        def scaling(x, y):
            if x < -1:
                fun = 1.0
                der1f = np.array([0.0, 0.0])
                der2f = np.zeros((2, 2))
                der3f = np.zeros((2, 2, 2))
                return fun, der1f, der2f, der3f
            elif x <= -0.5:
                xtilde = 2 * x + 2
                h, der1h, der2h, der3h = scaling_helper(xtilde)
                fun = -h * (1 - y) + 1
                der1f = np.array([-der1h * 2 * (1 - y), h])
                der2f = np.array([[-der2h * 4 * (1 - y), der1h * 2], [der1h * 2, 0]])
                der3f = np.zeros((2, 2, 2))
                der3f[0, 0, 0] = -der3h * 8 * (1 - y)
                der3f[0, 0, 1] = der3f[0, 1, 0] = der3f[1, 0, 0] = der2h * 4
                return fun, der1f, der2f, der3f
            elif x < 0.5:
                fun = y
                der1f = np.array([0.0, 1.0])
                der2f = np.zeros((2, 2))
                der3f = np.zeros((2, 2, 2))
                return fun, der1f, der2f, der3f
            elif x <= 1:
                xtilde = 2 * x - 1
                h, der1h, der2h, der3h = scaling_helper(xtilde)
                fun = h * (1 - y) + y
                der1f = np.array([der1h * 2 * (1 - y), -h + 1])
                der2f = np.array([[der2h * 4 * (1 - y), -der1h * 2], [-der1h * 2, 0]])
                der3f = np.zeros((2, 2, 2))
                der3f[0, 0, 0] = der3h * 8 * (1 - y)
                der3f[0, 0, 1] = der3f[0, 1, 0] = der3f[1, 0, 0] = -der2h * 4
                return fun, der1f, der2f, der3f
            else:
                fun = 1.0
                der1f = np.array([0.0, 0.0])
                der2f = np.zeros((2, 2))
                der3f = np.zeros((2, 2, 2))
                return fun, der1f, der2f, der3f

        def step(x, y):
            sig, der1sig, der2sig, der3sig = sigmoid_fun(y)
            s, der1s, der2s, der3s = scaling(x, 2 * sig)
            w, der1w, der2w, der3w = wiggly(x)
            fun = w * s
            der1f = np.zeros(2)
            der1f[0] = der1w * s + w * der1s[0]
            der1f[1] = w * der1s[1] * 2 * der1sig
            der2f = np.zeros((2, 2))
            der2f[0, 0] = der2w * s + 2 * der1w * der1s[0] + w * der2s[0, 0]
            der2f[0, 1] = (der1w * der1s[1] + w * der2s[0, 1]) * 2 * der1sig
            der2f[1, 0] = der2f[0, 1]
            der2f[1, 1] = w * (der2s[1, 1] * 4 * der1sig**2 + der1s[1] * 2 * der2sig)
            der3f = np.zeros((2, 2, 2))
            der3f[0, 0, 0] = der3w * s + 3 * der2w * der1s[0] + 3 * der1w * der2s[0, 0] + w * der3s[0, 0, 0]
            der3f[0, 0, 1] = (der2w * der1s[1] + 2 * der1w * der2s[0, 1] + w * der3s[0, 0, 1]) * 2 * der1sig
            der3f[0, 1, 0] = der3f[0, 0, 1]
            der3f[0, 1, 1] = der1w * (der2s[1, 1] * 4 * der1sig**2 + der1s[1] * 2 * der2sig) + w * (der3s[0, 1, 1] * 4 * der1sig**2 + der2s[0, 1] * 2 * der2sig)
            der3f[1, 0, 0] = der3f[0, 0, 1]
            der3f[1, 0, 1] = der3f[0, 1, 1]
            der3f[1, 1, 0] = der3f[0, 1, 1]
            der3f[1, 1, 1] = w * (der3s[1, 1, 1] * 8 * der1sig**3 + 3 * der2s[1, 1] * 4 * der1sig * der2sig + der1s[1] * 2 * der3sig)
            return fun, der1f, der2f, der3f

        def barrier(x, x_min, x_max):
            if x <= x_min:
                return (x - x_min) ** 4, 4 * (x - x_min) ** 3, 12 * (x - x_min) ** 2, 24 * (x - x_min)
            elif x >= x_max:
                return (x - x_max) ** 4, 4 * (x - x_max) ** 3, 12 * (x - x_max) ** 2, 24 * (x - x_max)
            else:
                return 0.0, 0.0, 0.0, 0.0

        def fX(X, Y):
            return np.zeros_like(X)

        def fx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            fun, _, _, _ = step(x, y)
            barrier_mult = 50.0
            barrier_box = [-0.4, 0.5, -5.0, 0.0]
            b_x, _, _, _ = barrier(x, barrier_box[0], barrier_box[1])
            b_y, _, _, _ = barrier(y, barrier_box[2], barrier_box[3])
            fun = fun + barrier_mult * b_x + barrier_mult * b_y + 3e-4 * x
            return fun

        def dx(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            _, der1f, _, _ = step(x, y)
            barrier_mult = 50.0
            barrier_box = [-0.4, 0.5, -5.0, 0.0]
            _, db_x, _, _ = barrier(x, barrier_box[0], barrier_box[1])
            _, db_y, _, _ = barrier(y, barrier_box[2], barrier_box[3])
            der1f[0] = der1f[0] + barrier_mult * db_x + 3e-4
            der1f[1] = der1f[1] + barrier_mult * db_y
            return der1f.reshape(-1, 1)

        def d2x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            _, _, der2f, _ = step(x, y)
            barrier_mult = 50.0
            barrier_box = [-0.4, 0.5, -5.0, 0.0]
            _, _, d2b_x, _ = barrier(x, barrier_box[0], barrier_box[1])
            _, _, d2b_y, _ = barrier(y, barrier_box[2], barrier_box[3])
            der2f[0, 0] = der2f[0, 0] + barrier_mult * d2b_x
            der2f[1, 1] = der2f[1, 1] + barrier_mult * d2b_y
            return der2f

        def d3x(X):
            x = X[0] if len(X.shape) == 1 else X[0, 0]
            y = X[1] if len(X.shape) == 1 else X[1, 0]
            _, _, _, der3f = step(x, y)
            barrier_mult = 50.0
            barrier_box = [-0.4, 0.5, -5.0, 0.0]
            _, _, _, d3b_x = barrier(x, barrier_box[0], barrier_box[1])
            _, _, _, d3b_y = barrier(y, barrier_box[2], barrier_box[3])
            der3f[0, 0, 0] = der3f[0, 0, 0] + barrier_mult * d3b_x
            der3f[1, 1, 1] = der3f[1, 1, 1] + barrier_mult * d3b_y
            return der3f

    elif func_name.startswith("Rosenbrock"):
        # Rosenbrock-n function: f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
        # Extract dimension from name (e.g., "Rosenbrock-20" -> n=20)
        try:
            n = int(func_name.split("-")[1]) if "-" in func_name else 2
        except:
            n = 2

        def fx(X):
            # X can be (n,) or (n, 1)
            x = X.flatten() if len(X.shape) > 1 else X
            n_actual = len(x)
            f_val = 0.0
            for i in range(n_actual - 1):
                f_val += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            return f_val

        def dx(X):
            # Gradient: df/dx_i
            x = X.flatten() if len(X.shape) > 1 else X
            n_actual = len(x)
            grad = np.zeros(n_actual)
            for i in range(n_actual - 1):
                # Term 1: 100(x_{i+1} - x_i^2)^2 contributes to x_i and x_{i+1}
                grad[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2)
                grad[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
                # Term 2: (1 - x_i)^2 contributes only to x_i
                grad[i] += -2 * (1 - x[i])
            return grad.reshape(-1, 1)

        def d2x(X):
            # Hessian: d^2f/dx_i dx_j
            x = X.flatten() if len(X.shape) > 1 else X
            n_actual = len(x)
            hess = np.zeros((n_actual, n_actual))
            for i in range(n_actual - 1):
                # Diagonal terms
                # d^2/dx_i^2 from 100(x_{i+1} - x_i^2)^2
                hess[i, i] += -400 * (x[i + 1] - x[i] ** 2) + 800 * x[i] ** 2
                # d^2/dx_i^2 from (1 - x_i)^2
                hess[i, i] += 2
                # d^2/dx_{i+1}^2 from 100(x_{i+1} - x_i^2)^2
                hess[i + 1, i + 1] += 200
                # Off-diagonal terms
                hess[i, i + 1] += -400 * x[i]
                hess[i + 1, i] += -400 * x[i]
            return hess

        def d3x(X):
            # Third-order tensor: d^3f/dx_i dx_j dx_k
            # This is sparse - only certain entries are non-zero
            x = X.flatten() if len(X.shape) > 1 else X
            n_actual = len(x)
            T = np.zeros((n_actual, n_actual, n_actual))
            for i in range(n_actual - 1):
                # Only non-zero entries involve x_i and x_{i+1}
                # d^3/dx_i^3 from 100(x_{i+1} - x_i^2)^2
                T[i, i, i] += 2400 * x[i]
                # d^3/dx_i^2 dx_{i+1} and permutations
                T[i, i, i + 1] += -400
                T[i, i + 1, i] += -400
                T[i + 1, i, i] += -400
            return T

        # For Rosenbrock, fX is not used (it's a 2D plotting function)
        def fX(X, Y):
            # Not applicable for high-dimensional Rosenbrock
            return fx(np.array([X, Y]))

    else:
        # Unknown function name
        raise ValueError(
            f"Unknown function name: '{func_name}'. "
            f"Supported functions: Beale, Bohachevsky, Goldstein, Himmelblau, McCormick, Styblinski, Slalom, HairpinTurn, Rosenbrock-n"
        )

    return [fX, fx, dx, d2x, d3x]


def init_params(func_name):

    if func_name == "Beale":
        XMIN = 2
        XMAX = 3.5
        YMIN = 0
        YMAX = 1
        x_min = np.array([[3, 0.5]])

    elif func_name == "Bohachevsky":
        XMIN = -1
        XMAX = 1
        YMIN = -1
        YMAX = 1
        x_min = np.array([[0, 0.0]])

    elif func_name == "Goldstein":
        XMIN = -2
        XMAX = 2
        YMIN = -3
        YMAX = 1
        x_min = np.array([[0, -1]])

    elif func_name == "Himmelblau":
        XMIN = -5
        XMAX = 5
        YMIN = -5
        YMAX = 5
        x_min = np.array(
            [
                [3, 2],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )

    elif func_name == "McCormick":
        XMIN = -1.5
        XMAX = 4
        YMIN = -3
        YMAX = 4
        x_min = np.array([[-0.54719, -1.54719], [2.5944, 1.5944]])

    elif func_name == "Styblinski":
        XMIN = -6
        XMAX = 6
        YMIN = -6
        YMAX = 6
        x_min = np.array(
            [
                [-2.903534, -2.903534],
                [-2.903534, 2.7468],
                [2.7468, -2.903534],
                [2.7468, 2.7468],
            ]
        )

    elif func_name == "Slalom":
        XMIN = -2.0
        XMAX = 2.0
        YMIN = -2.0
        YMAX = 2.0
        # Starting point from Cartis paper
        x_min = np.array([[0.5, 0.0]])

    elif func_name == "HairpinTurn":
        XMIN = -1.0
        XMAX = 1.0
        YMIN = -1.0
        YMAX = 1.0
        # Starting point from Cartis paper
        x_min = np.array([[0.5, 0.0]])

    elif func_name.startswith("Rosenbrock"):
        # Rosenbrock-n: standard difficult starting point is [-1.2, 1.0, ..., 1.0]
        try:
            n = int(func_name.split("-")[1]) if "-" in func_name else 2
        except:
            n = 2
        XMIN = -2.0
        XMAX = 2.0
        YMIN = -2.0
        YMAX = 2.0
        # Standard difficult starting point
        x0_standard = np.zeros(n)
        x0_standard[0] = -1.2
        x0_standard[1:] = 1.0
        x_min = x0_standard.reshape(1, -1)

    return [XMIN, XMAX, YMIN, YMAX, x_min]
