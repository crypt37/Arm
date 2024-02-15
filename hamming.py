import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 17

# Define the parameters for the Hamming window
window_length = 64  # Length of the window
alpha = 0.54        # Hamming window parameter
beta = 0.46         # Hamming window parameter

# Generate the Hamming window
n = np.arange(0, window_length)
hamming_window = alpha - beta * np.cos(2 * np.pi * n / (window_length - 1))

# Generate the frequency response of the Hamming window
fft_size = 1024  # Size of the FFT
hamming_window_freq = np.fft.fft(hamming_window, fft_size)
frequency = np.fft.fftfreq(fft_size)

# Peak to peak side lobe Amplitude (Relative)
peak_to_peak_side_lobe_amplitude = np.max(hamming_window_freq[1:]) - np.min(hamming_window_freq[1:])

# Find the width of the main lobe (Frequency Bins)
threshold = np.max(hamming_window_freq) / 2
main_lobe_width = np.sum(np.abs(hamming_window_freq) >= threshold)

# Peak Approximation Error
peak_approximation_error = np.max(hamming_window_freq) - 1

# Roll off factor (dB/octave)
roll_off_factor = -20 * np.log10(np.abs(hamming_window_freq[fft_size // 4]))

# Plot the Hamming window
plt.figure(figsize=(35, 40))

# Subplot 1: Hamming Window
plt.subplot(131)
plt.plot(n, hamming_window, 'b')
plt.title('Hamming Window')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)

# Subplot 2: Frequency Response
plt.subplot(132)
plt.plot(frequency, 20 * np.log10(np.abs(hamming_window_freq)))
plt.title('Normalized Frequency vs. Magnitude (dB)')
plt.xlabel('Normalized Frequency')
plt.xlim(-0.3, 0.3)
plt.ylim(-60, 60)
plt.ylabel('Magnitude (dB)')
plt.grid(True)

# Subplot 3: Parameter Legends
plt.subplot(133)
plt.text(0.1, 0.9, 'Parameter Values:', fontsize=14)
plt.text(0.1, 0.8, 'Hop length :' + f'20ms', fontsize=12)
plt.text(0.1, 0.7, 'Window length :' + f'20ms', fontsize=12)
plt.text(0.1, 0.6, 'Overlap length :' + f'10ms', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()
