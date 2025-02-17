import numpy as np
from scipy.integrate import solve_ivp


class BasePendulum:
    """
    Base class representing a simple pendulum, with or without damping.

    Parameters:
    - g: gravitational acceleration (m/s²)
    - L: pendulum length (m)
    - m: pendulum mass (kg)
    - k: viscous damping coefficient (kg/s)
    """

    def __init__(self, g=9.81, L=1.0, m=1.0, k=1.0):
        self.g = g
        self.L = L
        self.m = m
        self.k = k

    def _motion_equation(self, t, y):
        """
        Defines the differential equation governing the pendulum's motion.

        Parameters:
        - t: time (s)
        - y: state vector [theta, omega]

        Returns:
        - [dtheta/dt, domega/dt]
        """
        theta, omega = y
        dtheta_dt = omega
        domega_dt = - (self.g / self.L) * np.sin(theta) - (self.k / (self.m * self.L ** 2)) * omega
        return [dtheta_dt, domega_dt]

    def solve(self, t_span=(0.0, 10.0), theta0=np.pi / 4, omega0=0.0, dt=0.1):
        """
        Solves the motion equation over a given time interval.

        Parameters:
        - t_span: tuple (t0, tf) defining the time interval (s)
        - theta0: initial condition for the angle (rad)
        - omega0: initial condition for the angular velocity (rad/s)
        - dt: time step for solution evaluation (s)

        Returns:
        - t: array of time instances (s)
        - y: array of solutions [theta, omega]
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        sol = solve_ivp(fun=self._motion_equation, t_span=t_span, y0=[theta0, omega0], t_eval=t_eval, method="RK45")
        return sol.t, sol.y


class Pendulum(BasePendulum):
    """
    Implementation of a simple pendulum without damping.
    """

    def __init__(self, g=9.81, L=1.0, m=1.0):
        super().__init__(g, L, m, k=0.0)  # k=0 for an ideal frictionless pendulum


class DampedPendulum(BasePendulum):
    """
    Implementation of a damped pendulum with viscous friction.
    """

    def __init__(self, g=9.81, L=1.0, m=1.0, k=1.0):
        super().__init__(g, L, m, k)
