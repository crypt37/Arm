import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
import librosa.display
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
from matplotlib import rcParams
audio_path = '/home/neko/new/records/fetchaglassofwater.wav'

signal, sample_rate = librosa.load(audio_path)
rcParams['font.size'] = 16
# Calculate original signal length
original_length = len(signal) / sample_rate
print("Original Signal Length: {:.2f} seconds".format(original_length))

# Create a figure and set up the subplots
fig, axs = plt.subplots(5, 2, figsize=(10, 20),dpi=100)

# Plot original signal
axs[0, 0].plot(np.arange(len(signal)) / sample_rate, signal)
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].set_title('Original Signal')

# Plot spectrogram of the original signal
spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
axs[0, 1].imshow(spectrogram, origin='lower', aspect='auto', cmap='binary', extent=[0, original_length, 0, sample_rate / 2])
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Frequency (Hz)')
axs[0, 1].set_title('Spectrogram of Original Signal')

# Add random noise to the signal
noise_factor = 0.5
noise = np.random.normal(0, signal.std(), signal.size)
augmented_signal = signal + noise * noise_factor

# Calculate augmented signal length
augmented_length = len(augmented_signal) / sample_rate
print("Augmented Signal Length: {:.2f} seconds".format(augmented_length))

# Plot augmented signal
axs[1, 0].plot(np.arange(len(augmented_signal)) / sample_rate, augmented_signal)
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Amplitude')
axs[1, 0].set_title('Augmented Signal')

# Stretch the signal
stretch_rate = 0.75
stretched_signal, sample_rate = librosa.load(audio_path, sr=int(sample_rate * stretch_rate))

# Calculate stretched signal length
stretched_length = len(stretched_signal) / (sample_rate * stretch_rate)
print("Stretched Signal Length: {:.2f} seconds".format(stretched_length))

# Plot stretched signal
axs[1, 1].plot(np.arange(len(stretched_signal)) / (sample_rate * stretch_rate), stretched_signal)
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Amplitude')
axs[1, 1].set_title('Stretched Signal')

# Shift the pitch of the signal
num_semitones = 4
pitch_shifted_signal = librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=num_semitones)

# Calculate pitch-shifted signal length
pitch_shifted_length = len(pitch_shifted_signal) / sample_rate
print("Pitch-Shifted Signal Length: {:.2f} seconds".format(pitch_shifted_length))

# Plot pitch-shifted signal
axs[2, 0].plot(np.arange(len(pitch_shifted_signal)) / sample_rate, pitch_shifted_signal)
axs[2, 0].set_xlabel('Time (s)')
axs[2, 0].set_ylabel('Amplitude')
axs[2, 0].set_title('Pitch-Shifted Signal')

# Invert the polarity of the signal
inverse_polarity = signal * -1

# Calculate inverted polarity signal length
inverse_polarity_length = len(inverse_polarity) / sample_rate
print("Inverse Polarity Signal Length: {:.2f} seconds".format(inverse_polarity_length))

# Plot inverted polarity signal
axs[2, 1].plot(np.arange(len(inverse_polarity)) / sample_rate, inverse_polarity)
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Amplitude')
axs[2, 1].set_title('Inverse Polarity')

# Apply random gain to the signal
min_gain_factor = 2
max_gain_factor = 4
random_gain_signal = signal * random.uniform(min_gain_factor, max_gain_factor)

# Calculate random gain signal length
random_gain_length = len(random_gain_signal) / sample_rate
print("Random Gain Signal Length: {:.2f} seconds".format(random_gain_length))

# Plot random gain signal
axs[3, 0].plot(np.arange(len(random_gain_signal)) / sample_rate, random_gain_signal)
axs[3, 0].set_xlabel('Time (s)')
axs[3, 0].set_ylabel('Amplitude')
axs[3, 0].set_title('Random Gain')
# ... (previous code remains unchanged) ...

# ... (previous code remains unchanged) ...
# ... (previous code remains unchanged) ...

# ... (previous code remains unchanged) ...

# Apply time masking
mask_time = 1
masked_signal_time = np.copy(augmented_signal)
print("the values are " ,len(masked_signal_time),mask_time*sample_rate)
mask_start = random.randint(0, len(masked_signal_time) - mask_time * sample_rate)
masked_signal_time[mask_start:mask_start + mask_time * sample_rate] = 0.0

# Apply frequency masking to the time-masked signal
mask_freq = 2
masked_spectrogram_time = librosa.feature.melspectrogram(y=masked_signal_time, sr=sample_rate)
masked_spectrogram_freq = np.copy(masked_spectrogram_time)
mask_freq_start = random.randint(0, masked_spectrogram_freq.shape[0] - mask_freq)
masked_spectrogram_freq[mask_freq_start:mask_freq_start + mask_freq, :] = 0.0




# Plot masked time signal
axs[4, 0].plot(np.arange(len(masked_signal_time)) / sample_rate, masked_signal_time)
axs[4, 0].set_xlabel('Time (s)')
axs[4, 0].set_ylabel('Amplitude')
axs[4, 0].set_title('Masked Signal (Time)')

# Plot masked time spectrogram
axs[3, 0].imshow(librosa.power_to_db(masked_spectrogram_time, ref=np.max), origin='lower', aspect='auto', cmap='binary', extent=[0, original_length, 0, sample_rate / 2])
axs[3, 0].set_xlabel('Time (s)')
axs[3, 0].set_ylabel('Frequency (Hz)')
axs[3, 0].set_title('Masked Spectrogram (Time)')
# Plot masked frequency spectrogram
axs[4, 1].imshow(librosa.power_to_db(masked_spectrogram_freq, ref=np.max), origin='lower', aspect='auto', cmap='binary', extent=[0, original_length, 0, sample_rate / 2])
axs[4, 1].set_xlabel('Time (s)')
axs[4, 1].set_ylabel('Frequency (Hz)')
axs[4, 1].set_title('Masked Spectrogram (Frequency)')


mask_freq = 2
masked_spectrogram_time = librosa.feature.melspectrogram(y=augmented_signal, sr=sample_rate)
masked_spectrogram_freq = np.copy(masked_spectrogram_time)
mask_freq_start = random.randint(0, masked_spectrogram_freq.shape[0] - mask_freq)
masked_spectrogram_freq[mask_freq_start:mask_freq_start + mask_freq, :] = 0.0

# Plot masked frequency signal
axs[3, 1].imshow(librosa.power_to_db(masked_spectrogram_freq, ref=np.max), origin='lower', aspect='auto', cmap='binary', extent=[0, original_length, 0, sample_rate / 2])
axs[3, 1].set_xlabel('Time (s)')
axs[3, 1].set_ylabel('Amplitude')
axs[3, 1].set_title('Masked Signal (Frequency)')




# Adjust the spacing between subplots
plt.tight_layout()

# Save the plots to separate audio files
# sf.write("augmented_wave.flac", augmented_signal, sample_rate)
# sf.write("stretched_wave.flac", stretched_signal, sample_rate)
# sf.write("pitch_shifted.flac", pitch_shifted_signal, sample_rate)
# sf.write("inverse_polarity.flac", inverse_polarity, sample_rate)
# sf.write("random_gain.flac", random_gain_signal, sample_rate)
# sf.write("masked_time_wave.flac", masked_signal_time, sample_rate)
# sf.write("masked_freq_wave.flac", masked_signal_freq, sample_rate)

# Display the plot
plt.show()

