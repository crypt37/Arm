import numpy as np
import matplotlib.pyplot as plt

def hann_window(length):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))

# Define the length of the window
window_length = 100

# Generate the Hann window
window = hann_window(window_length)

# Plot the Hann window in the time domain
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(window)
plt.title('Hann Window (Time Domain)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Calculate the frequency domain representation (Fourier Transform)
freq_domain = np.fft.fft(window)

# Calculate the corresponding frequency values for the spectrum
sampling_rate = 2048.0  # In this example, we assume a sampling rate of 1 (arbitrary units)
frequencies = np.fft.fftfreq(window_length, d=1/sampling_rate)

# Shift the frequencies to center them around zero
frequencies_shifted = np.fft.fftshift(frequencies)
freq_domain_shifted = np.fft.fftshift(freq_domain)

# Plot the frequency domain representation (Magnitude Spectrum)
plt.subplot(1, 2, 2)
plt.plot(frequencies_shifted, np.abs(freq_domain_shifted))
plt.title('Frequency Domain Representation (Magnitude Spectrum)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()
