import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

import librosa.display
from scipy.signal import spectrogram

audio_path = 'sound.wav'

signal, sample_rate, = librosa.load(audio_path)
print(sample_rate)
frame_sie=2048
n_fft = 1024
hop_length = 512
n_mels = 80

#
# plt.subplot(4, 1, 1)
# plt.plot(signal)
#
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title(f'Amplitude Normalized Waveform for  {audio_path[4:-4]}')
# plt.grid(True)
# #
# Compute the spectrogram using stft

# plt.subplot(4, 1, 2)
# freq, time, spectrogram_data = spectrogram(signal, fs=sample_rate)
#
# # Plot the spectrogram
# plt.pcolormesh(time, freq, 10 * np.log10(spectrogram_data), shading='auto')


# # Compute and plot the Mel spect....................................................................rogram

frequencies, times, spectrogram = librosa.sft(signal,n_fft=frame_size,hop_length=hop_length,window='hann')

# Plot the spectrogram
print(frequencies.shape,times.shape)
print(frequencies)
plt.pcolormesh(times, frequencies, 10 * np.log10(np.abs(spectrogram)), shading='auto')
plt.colorbar(label='Amplitude (dB)')
plt.ylim(0,2000)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of "Bring Me a Glass of Water"')


mel_spec = librosa.feature.melspectrogram(S=np.abs(spectrogram), sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# plt.subplot(4, 1, 3)

# librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
#
# plt.title('Mel Spectrogram of "Bring Me a Glass of Water"')
# plt.xlabel('Time (s)')
# plt.ylabel('frequency(Hz)')
# print("the shape of spectogram" , spectrogram.shape)

# librosa.display.specshow(mel_spec_db, x_axis='time', sr=sample_rate)
# plt.title('MFCCs (dB) of "Bring Me a Glass of Water"')
# plt.xlabel('Time (s)')
# plt.ylabel('MFCC Coefficients')
# mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sample_rate/2)
#
# y_ticks = np.linspace(0, n_mels-1, 8)
# y_tick_labels = np.linspace(0, 8, 8, dtype=int)
# plt.yticks(y_ticks, y_tick_labels)
#
# plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
