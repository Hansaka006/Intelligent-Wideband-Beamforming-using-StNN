import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 16  # Number of elements (16-point DVM)
c = 3e8  # Speed of light (m/s)
fc = 2e9  # Center frequency (2 GHz)
BW = 1e9  # Bandwidth (1 GHz)
f_min = fc - BW / 2  # Lower frequency (1.5 GHz)
f_max = fc + BW / 2  # Upper frequency (2.5 GHz)
lambda_min = c / f_max  # Minimum wavelength
dx = lambda_min / 2  # Inter-element spacing

# Time delay per element
tau_unit = 2*dx / (c * N)

# Define spatio-temporal frequency axes
omega_x = np.linspace(-np.pi, np.pi, 100)  # Normalized spatial frequency
omega_t = np.linspace(-2 * np.pi * BW, 2 * np.pi * BW, 100)  # Normalized temporal frequency

# Beam steering angles (not used for titles now)
l_values = np.arange(0, N)  # Beam index

# Compute beamformer response for each beam direction
H_total = np.zeros((len(omega_x), len(omega_t), len(l_values)), dtype=complex)

for l in range(len(l_values)):
    tau_l = tau_unit * l_values[l]
    for k in range(N):
        H_total[:, :, l] += np.exp(-1j * k * (omega_x[:, None] + omega_t * tau_l))

# Visualization
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for l in range(len(l_values)):
    ax = axes.flatten()[l]
    im = ax.imshow(np.abs(H_total[:, :, l]), aspect='auto', extent=[omega_t[0] / (2 * np.pi * BW), omega_t[-1] / (2 * np.pi * BW), omega_x[-1] / np.pi, omega_x[0] / np.pi], cmap='jet')
    ax.set_xlabel(r'$\omega_t / (2\pi BW)$')
    ax.set_ylabel(r'$\omega_x / \pi$')
    ax.set_title(f'Bin {l + 1}')
    fig.colorbar(im, ax=ax)

plt.suptitle('Spatio-Temporal Frequency Response of 16-Point DVM Beamformer')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
