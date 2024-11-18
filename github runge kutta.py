import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Constants
G = 6.6726e-11        # Gravitational constant (m^3 kg^-1 s^-2)
m_earth = 5.9742e24    # Mass of Earth (kg)
m_moon = 7.35e22       # Mass of Moon (kg)
M = m_earth + m_moon   # Total mass (kg)
D = 3.844e8            # Average Earth-Moon distance (m)
mu = m_moon / M        # Mass ratio
omega = np.sqrt(G * M / D**3)
x_earth = -mu * D
y_earth = 0.0
x_moon = (1 - mu) * D
y_moon = 0.0

# Function to find L2 point with higher precision
def f_L2(x):
    r1 = x - x_moon
    r2 = x - x_earth
    return -G * m_moon / r1**2 - G * m_earth / r2**2 + omega**2 * x

x_L2 = brentq(f_L2, x_moon + 1e5, x_moon + D, xtol=1e-12)

# Function to compute initial vy0
def compute_vy0(x0):
    y0 = 0.0
    vx0 = 0.0
    r1 = np.sqrt((x0 - x_moon)**2 + y0**2)
    r2 = np.sqrt((x0 - x_earth)**2 + y0**2)
    ax_gravity = (-G * m_moon * (x0 - x_moon) / r1**3
                  - G * m_earth * (x0 - x_earth) / r2**3)
    ax_centrifugal = omega**2 * x0
    ax_total = ax_gravity + ax_centrifugal
    vy0 = -ax_total / (2 * omega)
    return vy0

# Function to compute accelerations
def accelerations(x, y, vx, vy):
    dx_earth = x - x_earth
    dy_earth = y - y_earth
    r_earth = np.sqrt(dx_earth**2 + dy_earth**2)
    dx_moon = x - x_moon
    dy_moon = y - y_moon
    r_moon = np.sqrt(dx_moon**2 + dy_moon**2)
    ax = (-G * m_earth * dx_earth / r_earth**3
          - G * m_moon * dx_moon / r_moon**3
          + omega**2 * x
          + 2 * omega * vy)
    ay = (-G * m_earth * dy_earth / r_earth**3
          - G * m_moon * dy_moon / r_moon**3
          + omega**2 * y
          - 2 * omega * vx)
    return ax, ay

# Runge-Kutta method
def runge_kutta_method(x0, vy0, dt, total_time):
    t = np.arange(0, total_time + dt, dt)
    num_steps = len(t)
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    vx = np.zeros(num_steps)
    vy = np.zeros(num_steps)
    x[0] = x0
    y[0] = 0.0
    vx[0] = 0.0
    vy[0] = vy0
    for n in range(num_steps - 1):
        k1x = vx[n]
        k1y = vy[n]
        k1vx, k1vy = accelerations(x[n], y[n], vx[n], vy[n])

        x_k2 = x[n] + 0.5 * dt * k1x
        y_k2 = y[n] + 0.5 * dt * k1y
        vx_k2 = vx[n] + 0.5 * dt * k1vx
        vy_k2 = vy[n] + 0.5 * dt * k1vy

        k2x = vx_k2
        k2y = vy_k2
        k2vx, k2vy = accelerations(x_k2, y_k2, vx_k2, vy_k2)

        x_k3 = x[n] + 0.5 * dt * k2x
        y_k3 = y[n] + 0.5 * dt * k2y
        vx_k3 = vx[n] + 0.5 * dt * k2vx
        vy_k3 = vy[n] + 0.5 * dt * k2vy

        k3x = vx_k3
        k3y = vy_k3
        k3vx, k3vy = accelerations(x_k3, y_k3, vx_k3, vy_k3)

        x_k4 = x[n] + dt * k3x
        y_k4 = y[n] + dt * k3y
        vx_k4 = vx[n] + dt * k3vx
        vy_k4 = vy[n] + dt * k3vy

        k4x = vx_k4
        k4y = vy_k4
        k4vx, k4vy = accelerations(x_k4, y_k4, vx_k4, vy_k4)

        x[n+1] = x[n] + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        y[n+1] = y[n] + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
        vx[n+1] = vx[n] + (dt / 6.0) * (k1vx + 2*k2vx + 2*k3vx + k4vx)
        vy[n+1] = vy[n] + (dt / 6.0) * (k1vy + 2*k2vy + 2*k3vy + k4vy)

    return x, y

# Parameters
T = 2 * np.pi / omega             # Orbital period
dt = 10                            # Time step
total_time = T                     # Total simulation time
num_trials = 10                     # Number of trials
percentage_variation = 0.001        # Variation percentage

desired_distance_from_moon = 64520110.0  # Desired initial distance from the Moon (m)
x0_base = x_moon + desired_distance_from_moon
x0_min = x0_base * (1 - percentage_variation)
x0_max = x0_base * (1 + percentage_variation)

if num_trials == 1:
    x0_trials = np.array([x0_base])
else:
    x0_trials = np.linspace(x0_min, x0_max, num_trials)
    closest_idx = np.argmin(np.abs(x0_trials - x0_base))
    x0_trials[closest_idx] = x0_base

min_max_deviation = np.inf
best_delta_r = None
best_x = None
best_y = None
best_trial_index = None

for idx, x0_trial in enumerate(x0_trials):
    print(f"We are doing Trial {idx + 1} at {x0_trial} m")
    vy0 = compute_vy0(x0_trial)
    x_traj, y_traj = runge_kutta_method(x0_trial, vy0, dt, total_time)
    deviations = np.sqrt((x_traj - x_L2)**2 + y_traj**2)
    idx_one_orbit = int(T / dt)
    if idx_one_orbit < len(deviations):
        separation_after_one_orbit = deviations[idx_one_orbit]
    else:
        separation_after_one_orbit = deviations[-1]
    max_deviation = np.max(deviations)
    if max_deviation < min_max_deviation:
        min_max_deviation = max_deviation
        best_delta_r = x0_trial - x_L2
        best_x = x_traj
        best_y = y_traj
        best_separation = separation_after_one_orbit
        best_trial_index = idx

optimal_starting_distance = best_x[0] - x_moon
optimal_starting_distance_km = optimal_starting_distance / 1e3
best_separation_km = best_separation / 1e3

print(f"\nThe optimal starting point for L2 from the Moon is: {optimal_starting_distance_km:.5f} km")
print(f"The distance after 1 lunar orbit from the Moon is: {best_separation_km:.5f} km")

# Plotting parameters
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Time array
t = np.arange(0, total_time + dt, dt)

# Compute inertial frame positions
x_moon_inertial = x_moon * np.cos(omega * t)
y_moon_inertial = x_moon * np.sin(omega * t)
x_sat_inertial = best_x * np.cos(omega * t) - best_y * np.sin(omega * t)
y_sat_inertial = best_x * np.sin(omega * t) + best_y * np.cos(omega * t)

# Plot trajectories in inertial frame
plt.figure()
plt.title('Runge-Kutta Method: Trajectories of Moon and Satellite in Inertial Frame')
plt.xlabel('X Position (km)')
plt.ylabel('Y Position (km)')
plt.axis('equal')

x_moon_inertial_km = x_moon_inertial / 1e3
y_moon_inertial_km = y_moon_inertial / 1e3
x_sat_inertial_km = x_sat_inertial / 1e3
y_sat_inertial_km = y_sat_inertial / 1e3

plt.plot(0, 0, 'bo', markersize=12, label='Earth')
plt.plot(x_moon_inertial_km, y_moon_inertial_km, 'r-', linewidth=2, label='Moon Trajectory')
plt.plot(x_sat_inertial_km, y_sat_inertial_km, 'k-', linewidth=2, label='Satellite Trajectory')
plt.plot(x_moon_inertial_km[0], y_moon_inertial_km[0], 'ro', markersize=8, label='Moon Start')
plt.plot(x_sat_inertial_km[0], y_sat_inertial_km[0], 'go', markersize=8, label='Satellite Start')
plt.legend(loc='upper right')
plt.grid(False)
plt.show()
