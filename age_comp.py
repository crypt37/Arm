import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
import librosa.display
import librosa
from matplotlib import rcParams
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
rcParams['font.size'] = 15

# Define a function to plot signal, spectrogram, and histogram
def plot_signal_spectrogram_histogram(signal, augmented_signal, spectrogram, title, aug_name):
    # Plot original signal
    axs[0, 0].plot(np.arange(len(signal)) / sample_rate, signal,color='blue')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].set_title('Original Signal')
    # augmented_spectrogram = librosa.feature.melspectrogram(y=augmented_signal, sr=sample_rate)
    # augmented_spectrogram = librosa.power_to_db(augmented_spectrogram, ref=np.max)

    spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
    spec = librosa.power_to_db(spec, ref=np.max)
    # Plot spectrogram of the original signal
    axs[0, 1].imshow(spec, origin='lower', aspect='auto', cmap='binary', extent=[0, original_length, 0, sample_rate / 2])
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Frequency (Hz)')
    axs[0, 1].set_title('Spectrogram of Original Signal')

    # reconstructed_signal = librosa.feature.inverse.mel_to_audio(librosa.power_to_db(augmented_signal))
    recovered_signal = librosa.istft(augmented_signal)

    # Plot augmented signal
    axs[1, 0].plot(np.arange(len(recovered_signal)) / sample_rate, recovered_signal,color='blue')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].set_title(title)

    # Calculate and plot spectrogram of the augmented signal

    axs[1, 1].imshow(librosa.power_to_db(augmented_signal), origin='lower', aspect='auto', cmap='binary', extent=[0, original_length, 0, sample_rate / 2])
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Frequency (Hz)')
    axs[1, 1].set_title(f'Spectrogram after {aug_name}')

    # Plot histogram of the augmented signal
    axs[2, 1].hist(recovered_signal, bins=100, range=(-0.005, 0.005), density=True, alpha=0.6,color='blue')


    axs[2, 1].set_ylim(0, 450)
    axs[2, 1].set_xlabel('Amplitude')
    axs[2, 1].set_ylabel('Density')
    axs[2, 1].set_title(f'Histogram after {aug_name}')
    # Plot histogram of the augmented signal

    axs[2, 0].hist(signal, bins=100, range=(-0.005, 0.005), density=True, alpha=0.6, color='blue')

    axs[2, 0].set_xlabel('Amplitude')


    axs[2, 0].set_ylabel('Density')
    axs[2, 0].set_title(f'Histogram of original signal')

# Load the audio signal
audio_path = '/media/hard-drive/nlp/nlp_water/water.wav'
signal, sample_rate = librosa.load(audio_path)

# Calculate original signal length
original_length = len(signal) / sample_rate
print("Original Signal Length: {:.2f} seconds".format(original_length))

# Create a figure and set up the subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))


# Apply random noise to the signal
noise_factor = 0.5
noise = np.random.normal(0, signal.std(), signal.size)
augmented_signal = signal + noise * noise_factor

# Apply time stretching
stretch_rate = 0.75
stretched_signal, sample_rate = librosa.load(audio_path, sr=int(sample_rate * stretch_rate))

# Shift the pitch of the signal
num_semitones = 4
pitch_shifted_signal = librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=num_semitones)

# Invert the polarity of the signal
inverse_polarity = signal * -1

# Apply random gain to the signal
min_gain_factor = 2
max_gain_factor = 4
random_gain_signal = signal * random.uniform(min_gain_factor, max_gain_factor)

# Apply time masking
mask_time = 5
masked_signal_time = np.copy(augmented_signal)
mask_start = random.randint(0, len(masked_signal_time) - mask_time * sample_rate)
masked_signal_time[mask_start:mask_start + mask_time * sample_rate] = 0.0

# Calculate spectrogram of the original signal
spectrogram = librosa.feature.melspectrogram(y=masked_signal_time, sr=sample_rate)
spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

mask_freq = 2
masked_spectrogram_time = librosa.feature.melspectrogram(y=masked_signal_time, sr=sample_rate)
masked_spectrogram_freq = np.copy(masked_spectrogram_time)
# mask_freq_start = random.randint(0, masked_spectrogram_freq.shape[0] - mask_freq)
# masked_spectrogram_freq[mask_freq_start:mask_freq_start + mask_freq, :] = 0.0

spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

plt.subplots_adjust(wspace=0.36, hspace=0.58)
# Plot all augmentations with titles
# plot_signal_spectrogram_histogram(signal, augmented_signal, spectrogram, 'Augmented Signal', 'Gaussian Noise')
# plt.show()
# plot_signal_spectrogram_histogram(signal, stretched_signal, spectrogram, 'Stretched Signal', 'Time Stretch')
# plt.show()
# plot_signal_spectrogram_histogram(signal, pitch_shifted_signal, spectrogram, 'Pitch-Shifted Signal', 'Pitch Shift')
# plt.show()
# plot_signal_spectrogram_histogram(signal, inverse_polarity, spectrogram, 'Inverse Polarity', 'Invert Polarity')
# plt.show()
# plot_signal_spectrogram_histogram(signal, random_gain_signal, spectrogram, 'Random Gain', 'Random Gain')

# plot_signal_spectrogram_histogram(signal, masked_spectrogram_time, spectrogram, 'Masked Signal (Time)', 'Time Mask')
# # plot_signal_spectrogram_histogram(signal, masked_spectrogram_time, spectrogram, 'Masked Signal (Time)', 'Time Mask')


plt.show()
# Adjust the spacing between subplots

plt.tight_layout()

# Display the plots
plt.show()
