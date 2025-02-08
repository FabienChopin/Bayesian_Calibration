import numpy as np
from scipy.integrate import solve_ivp

class BasePendulum:
    """Classe de base pour un pendule, avec ou sans amortissement."""
    def __init__(self, g=9.81, L=1., m=1., k=1.):
        self.g = g # Accélération de la pesanteur (m/s^2)
        self.L = L # longueur du pendule (m)
        self.m = m # Masse de l'objet (kg)
        self.k = k # Coefficient de frottement visqueux (kg/s)

    def ode(self, t, y):
        """Équation du mouvement du pendule."""
        theta, omega = y
        dtheta_dt = omega
        domega_dt = - (self.g / self.L) * np.sin(theta)- (self.k / (self.m * self.L**2)) * omega
        return [dtheta_dt, domega_dt]

    def solve(self, t_span=(0, 10), theta0=np.pi/4, omega0=0., dt=0.01):
        """Résout l'ODE pour une plage de temps donnée."""
        t_eval = np.arange(t_span[0], t_span[1], dt)
        sol = solve_ivp(fun=self.ode, t_span=t_span, y0=[theta0, omega0], t_eval=t_eval, method='RK45')
        return sol.t, sol.y


class Pendulum(BasePendulum):
    """Pendule simple (sans amortissement)."""

    def __init__(self, g=9.81, L=1., m=1.):
        super().__init__(g, L, m, k=0.)  # k=0 pour un pendule sans frottement


class DampedPendulum(BasePendulum):
    """Pendule amorti."""

    def __init__(self, g=9.81, L=1., m=1., k=1.):
        super().__init__(g, L, m, k)