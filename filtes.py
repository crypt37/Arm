import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal
from matplotlib import rcParams
rcParams['font.size'] = 17
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
audio_path = '/media/hard-drive/nlp/flac/flac/34-586334-01.flac'
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


gain_original, phase_original = calculate_gain_and_phase(signal_wav)
gain_filtered, phase_filtered = calculate_gain_and_phase(filtered_signal_band)


# Create subplots for original and filtered signals, gain, and phase
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Plot the original and filtered signals
axs[0, 0].plot(np.arange(len(signal_wav)) / sample_rate, signal_wav)
axs[0, 0].set_title('Original Signal')
axs[1, 0].plot(np.arange(len(filtered_signal_band)) / sample_rate, filtered_signal_band)
axs[1, 0].set_title('Band-Pass Filtered Signal')

# Plot the gain of the original and filtered signals
axs[0, 1].plot(np.arange(len(gain_original)) / sample_rate, gain_original)
axs[0, 1].set_title('Gain of Original Signal')
axs[1, 1].plot(np.arange(len(gain_filtered)) / sample_rate, gain_filtered)
axs[1, 1].set_title('Gain of Filtered Signal')

# Scale phase values to the range of -π to π radians
phase_original = np.angle(np.exp(1j * phase_original))
phase_filtered = np.angle(np.exp(1j * phase_filtered))

# Plot the phase of the original and filtered signals using polar projection
# Plot the phase in the second polar plot

# Calculate the mel spectrograms of the original and filtered signals
mel_spec_original = librosa.feature.melspectrogram(y=signal_wav, sr=sample_rate, n_mels=80)
log_mel_spec_original = librosa.power_to_db(mel_spec_original)
mel_spec_filtered = librosa.feature.melspectrogram(y=filtered_signal_band, sr=sample_rate, n_mels=80)
log_mel_spec_filtered = librosa.power_to_db(mel_spec_filtered)


# Plot the mel spectrograms
librosa.display.specshow(log_mel_spec_original, x_axis='time', y_axis='mel', sr=sample_rate, cmap="binary", ax=axs[2, 0])
axs[2, 0].set_title('Spectrogram of Original Signal')
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('Mel Frequency')

librosa.display.specshow(log_mel_spec_filtered, x_axis='time', y_axis='mel', sr=sample_rate, cmap="binary", ax=axs[2, 1])
axs[2, 1].set_title('Spectrogram of Filtered Signal')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Mel Frequency')

# Label axes
for ax in axs.ravel():
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude or Magnitude')

# Adjust subplot spacing
plt.tight_layout()
plt.show()
