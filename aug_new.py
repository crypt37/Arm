import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
import librosa.display
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter

audio_path = '/home/neko/new/records/fetchaglassofwater.wav'

signal, sample_rate = librosa.load(audio_path)

# Calculate original signal length
original_length = len(signal) / sample_rate
print("Original Signal Length: {:.2f} seconds".format(original_length))


# Function to plot signal and spectrogram with histogram
def plot_signal_spectrogram_histogram(signal, augmented_signal, title, augmentation_name):
    plt.figure(figsize=(10, 10))

    # Plot original signal
    plt.subplot(3, 2, 1)
    plt.plot(np.arange(len(signal)) / sample_rate, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original Signal - ' + title)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Calculate and plot histogram of original signal
    plt.subplot(3, 2, 2)
    plt.hist(signal, bins=100, range=(-1, 1), density=True, alpha=0.6)
    plt.xlabel('Amplitude')
    plt.ylabel('Probability Density')
    plt.title('Signal Histogram - ' + title)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Calculate and plot spectrogram of the original signal
    spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    plt.subplot(3, 2, 3)
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='mel', cmap="binary")
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.title('Spectrogram of Original Signal - ' + title)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Plot augmented signal
    plt.subplot(3, 2, 4)
    reconstructed_signal = librosa.feature.inverse.mel_to_audio(augmented_signal)

    plt.plot(np.arange(len(reconstructed_signal)) / sample_rate, reconstructed_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Augmented Signal - ' + title)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Calculate and plot histogram of augmented signal
    plt.subplot(3, 2, 5)
    plt.hist(reconstructed_signal, bins=100, range=(-1, 1), density=True, alpha=0.6)
    plt.xlabel('Amplitude')
    plt.ylabel('Probability Density')
    plt.title('Signal Histogram (Augmented) - ' + title)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Calculate and plot spectrogram of the augmented signal
    augmented_spectrogram = librosa.power_to_db(augmented_signal, ref=np.max)
    plt.subplot(3, 2, 6)
    librosa.display.specshow(augmented_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel', cmap="binary")
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.title('Spectrogram of Augmented Signal - ' + title)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Adjust the spacing between subplots
    plt.tight_layout()


# Add random noise to the signal
noise_factor = 0.5
noise = np.random.normal(0, signal.std(), signal.size)
augmented_signal_noise = signal + noise * noise_factor

# Stretch the signal
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
masked_signal_time = np.copy(augmented_signal_noise)
mask_start = random.randint(0, len(masked_signal_time) - mask_time * sample_rate)
masked_signal_time[mask_start:mask_start + mask_time * sample_rate+1] = 0.0
mask_freq = 2
masked_signal_time_freq = np.copy(augmented_signal_noise)
masked_spectrogram_time_freq = librosa.feature.melspectrogram(y=masked_signal_time_freq, sr=sample_rate)
mask_freq_start = random.randint(0, len(masked_signal_time_freq) - mask_freq * sample_rate)
masked_signal_time_freq[mask_freq_start:mask_freq_start + mask_freq * sample_rate] = 0.0

# Calculate and plot spectrogram of the frequency-masked signal

mask_freq = 2
masked_spectrogram_time = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
masked_spectrogram_freq = np.copy(masked_spectrogram_time)
mask_freq_start = random.randint(0, masked_spectrogram_freq.shape[0] - mask_freq)
masked_spectrogram_freq[mask_freq_start:mask_freq_start + mask_freq, :] = 0.0

plot_signal_spectrogram_histogram(signal, masked_spectrogram_freq,
                                  'Masked Signal (Frequency)', 'Frequency Mask')

# Display the plots
plt.show()
