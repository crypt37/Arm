import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal
from matplotlib import rcParams
rcParams['font.size'] = 12

# Function to apply a band-pass filter to the signal
def apply_band_pass_filter(signal_data, low_cutoff_freq, high_cutoff_freq, sampling_freq):
    nyquist_freq = 0.5 * sampling_freq
    normalized_low_cutoff = low_cutoff_freq / nyquist_freq
    normalized_high_cutoff = high_cutoff_freq / nyquist_freq

    # Design the Butterworth band-pass filter
    order = 4
    b, a = signal.butter(order, [normalized_low_cutoff, normalized_high_cutoff], btype='band')

    # Apply the filter to the signal
    filtered_signal = signal.lfilter(b, a, signal_data)

    return filtered_signal

# Function to calculate gain and phase
def calculate_gain_and_phase(signal_data):
    # Calculate the gain (magnitude) and phase of the signal
    gain = np.abs(signal_data)
    phase = np.angle(signal_data)

    return gain, phase

# Load the audio signal
audio_path = '/media/hard-drive/nlp/voice/fifty ft _FILL UP A CUP WITH ORANGE JUICE, _18.flac'
signal_wav, sample_rate = librosa.load(audio_path)

# Calculate the Fourier transform of the original signal
original_spectrum = np.fft.fft(signal_wav)

# Calculate the phase and magnitude of the original signal
phase_original = np.angle(original_spectrum)
magnitude_original = np.abs(original_spectrum)

# Define the frequency range for the band-pass filter
low_cutoff_freq = 90
high_cutoff_freq = 800

# Apply the band-pass filter to the signal
filtered_signal_band = apply_band_pass_filter(signal_wav, low_cutoff_freq, high_cutoff_freq, sample_rate)

# Calculate the Fourier transform of the filtered signal
filtered_spectrum = np.fft.fft(filtered_signal_band)

# Calculate the phase and magnitude of the filtered signal
phase_filtered = np.angle(filtered_spectrum)
magnitude_filtered = np.abs(filtered_spectrum)

# Create a polar plot for the phase and magnitude of the original and filtered signals
fig, axs = plt.subplots(2, 1, figsize=(8, 12), subplot_kw=dict(projection='polar'))

# Plot the magnitude in the first polar plot
magnitude_original_line, = axs[0].plot(np.angle(np.exp(1j * np.linspace(0, 2 * np.pi, len(magnitude_original)))), magnitude_original, label='Original Signal', color='gray')
magnitude_filtered_line, = axs[0].plot(np.angle(np.exp(1j * np.linspace(0, 2 * np.pi, len(magnitude_filtered)))), magnitude_filtered, label='Filtered Signal', color='black')
axs[0].set_title('Magnitude')

# Move the legend for the first plot outside the plot area
axs[0].legend(handles=[magnitude_original_line, magnitude_filtered_line], loc='upper right', bbox_to_anchor=(1.8, 1))




# Plot the phase in the second polar plot
axs[1].plot(np.angle(np.exp(1j * np.linspace(0, 2 * np.pi, len(phase_original)))), phase_original, label='Original Signal', color='black')
axs[1].plot(np.angle(np.exp(1j * np.linspace(0, 2 * np.pi, len(phase_filtered)))), phase_filtered, label='Filtered Signal', color='gray')
axs[1].set_title('Phase')
axs[1].legend()
plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1))


plt.tight_layout()
plt.show()
