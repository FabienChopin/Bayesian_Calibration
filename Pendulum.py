import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres standards du pendule
g = 9.81 # Accélération de la pesanteur (m/s^2)
L = 1.   # Longueur du pendule (m)
m = 1.   # Masse de l'objet (kg)
k = 1.   # Coefficient de frottement visqueux (kg/s)


# Equation régissant le pendule
def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta) - (k / (m * L**2)) * omega
    return [dtheta_dt, domega_dt]

# Conditions initiales
t_init, t_final = 0, 10
y_init = [np.pi / 4, 0]  # Position et vitesse initiale (rad et rad/s)
nombre_pas_de_temps = 100
t_range = np.linspace(t_init, t_final, nombre_pas_de_temps)

# Résolution de l'EDO
sol_standard = solve_ivp(pendulum, [t_init, t_final], y_init, t_eval=t_range, method='RK45')

# Tracé des résultats
plt.figure(figsize=(8, 6))

nombre_cas = 10000
resultats = np.empty((nombre_pas_de_temps * nombre_cas, 6))

for i in range(nombre_cas):
  y_init = [np.random.uniform(-0.5, 0.5) * np.pi, 0]
  L = np.random.uniform(0.5, 1.5)
  k = np.random.uniform(0.5, 1.5)
  m = np.random.uniform(0.8, 1.2)
  param = np.repeat([[y_init[0], L, k, m]], 100, axis=0)
  sol = solve_ivp(pendulum, [t_init, t_final], y_init, t_eval=t_range, method='RK45')
  temps = sol.t.reshape(-1, 1)
  angle = sol.y[0].reshape(-1, 1)
  y = np.concatenate((temps, angle), 1)
  param_et_y = np.concatenate((param, y), 1)
  resultats[i*nombre_pas_de_temps:(i+1)*nombre_pas_de_temps, :] = param_et_y
  plt.plot(sol.t, sol.y[0], color="C1")

plt.plot(sol_standard.t, sol_standard.y[0], color="C0", label='Angle (radians)')
plt.xlabel('Temps (s)')
plt.ylabel('Valeurs')
plt.title('Évolution du pendule avec amortissement')
plt.legend()
plt.grid()
plt.show()

print(resultats)
print(resultats.shape)