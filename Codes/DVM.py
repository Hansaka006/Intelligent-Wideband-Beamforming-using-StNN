import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 16  # Number of elements
fc = 27e9  # Carrier frequency (27 GHz)
f = 28e9
c = 3e8  # Speed of light
lamb = c / fc  # Wavelength at carrier frequency

l_min = c/f   # Wavelength at max frequency

dx = l_min / 2  # Inter-element spacing

# Compute 16x16 DVM matrix
tau_unit = 2 * dx / (c * N)  # Unit time delay
dvm_matrix = np.exp(-1j * 2 * np.pi * fc * np.outer(np.arange(N), tau_unit * np.arange(N)))  

# Time vector
t = np.linspace(0, 1e-6, 1000)  # 1 µs duration, 1000 samples

# Angle of arrival (AoA) - User-defined
theta_deg = 30  # Change this to test different angles
theta = np.deg2rad(theta_deg)  # Convert to radians

# phase shift calculation for input waveforms
phase_shifts = 2 * np.pi * np.arange(N) * dx / lamb * np.sin(theta)  

# Use complex exponential signals (instead of real cosines)
input_signals = np.array([np.exp(1j * (2 * np.pi * fc * t + phase_shifts[i])) for i in range(N)])


# Multiply input signals with DVM matrix to generate transmitted signals
transmitted_signals = np.dot(dvm_matrix, input_signals)

# ====== PLOT INPUT SIGNALS ======
fig1, axes1 = plt.subplots(4, 4, figsize=(12, 10))
fig1.suptitle("Input Signals (Before Beamforming)", fontsize=14)

for i in range(16):
    row, col = divmod(i, 4)  # Compute subplot position
    axes1[row, col].plot(t, np.real(input_signals[i]), color='r')  # Real part of input
    axes1[row, col].set_title(f"Input {i+1}", fontsize=10)
    axes1[row, col].set_xlabel("Time (s)", fontsize=8)
    axes1[row, col].set_ylabel("Amplitude", fontsize=8)
    axes1[row, col].grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
plt.show()

# ====== PLOT BEAMFORMED OUTPUTS ======
fig2, axes2 = plt.subplots(4, 4, figsize=(12, 10))
fig2.suptitle(f"Beamformed Signals for 16 Beams with AoA {theta_deg}°", fontsize=14)

for i in range(16):
    row, col = divmod(i, 4)  # Compute subplot position
    axes2[row, col].plot(t, np.real(transmitted_signals[i, :]), color='b')  # Real part
    axes2[row, col].set_title(f"Beam {i+1}", fontsize=10)
    axes2[row, col].set_xlabel("Time (s)", fontsize=8)
    axes2[row, col].set_ylabel("Amplitude", fontsize=8)
    axes2[row, col].grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
plt.show()
