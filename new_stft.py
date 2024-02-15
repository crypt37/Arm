import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

audio_path = '/home/neko/new/hey_arm/flac/18-568959-0002.flac'

signal, sample_rate = librosa.load(audio_path)
time = np.arange(0, len(signal)) / sample_rate
plt.figure(figsize=(16, 6))
plt.subplot(3, 1, 1)
plt.plot(time, signal)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Volts) ')
plt.title(f'Amplitude Normalized Waveform ')
plt.grid(True)

print("the sample rage of sample is ", signal.shape)
frame_size = 2048
n_fft = 2048
hop_length = 1024
n_mels = 80
n_mfccs = 13

s_scale = librosa.stft(y=signal, n_fft=frame_size, hop_length=hop_length)

print("sample after sft", s_scale.shape)
print(type(s_scale[0][0]))

Y_scale = np.abs(s_scale) ** 2
print(Y_scale.shape)

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(Y_scale, ref=np.max), sr=sample_rate, hop_length=hop_length,
                         x_axis='time', y_axis='log', cmap="binary")
colormap = "civids"
plt.colorbar(format='%+2.0f dB')
plt.title(f'Spectrogram ')
plt.ylabel("frequency(Hz)")
plt.xlabel('Time (s)')


# filter_banks = librosa.filters.mel(n_fft=n_fft, sr=sample_rate, n_mels=n_mels)
#
# librosa.display.specshow(filter_banks, sr=sample_rate, x_axis="linear", y_axis='mel', cmap='binary')
#
#
# plt.ylabel("Mel frequencies")
# plt.xlabel("Frequency (Hz)")
# plt.title("Mel Filter Banks")
#

mel_spec = librosa.feature.melspectrogram(S=np.abs(Y_scale), sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels)
print("mel spec shape ", mel_spec.shape)

mel_spec_db = librosa.power_to_db(mel_spec)

plt.subplot(3, 1, 3)
librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap="binary")
plt.xlabel("Time(s)")
plt.ylabel("frequency(Hz)")
plt.colorbar(format="%+2.f")
plt.title(f'Mel Spectrogram  representation')
plt.show()



mfccs_coeffs = librosa.feature.mfcc(y=signal,n_mfcc=n_mfccs,sr=sample_rate)
print(mfccs_coeffs.shape)

librosa.display.specshow(mfccs_coeffs, x_axis="linear", sr=sample_rate,cmap = "binary")
plt.xlabel("Time(s)")
plt.ylabel("Coefficients")
y_ticks = np.linspace(0, n_mfccs-1, 13)
y_tick_labels = np.linspace(0, 13, 13, dtype=int)
plt.yticks(y_ticks, y_tick_labels)
plt.colorbar(format="%+2.0f")  # Corrected format specifier
plt.title(f'MFCC Coefficients of "{audio_path[10:-4]}"')  # Corrected title
# print(np.array(mfccs_coeffs[:,0:5]))


plt.show()



# librosa.display.specshow(output_np, sr=sample_rate, x_axis="linear", y_axis='mel', cmap='binary')
plt.imshow(output_np[0, :, :, 0], cmap='binary')
plt.ylabel("Mel frequencies")
plt.xlabel("Frequency (Hz)")
plt.title("Mel spectogram after convolution and relu")
plt.show()


# plt.title('MFCCs (dB) of "Bring Me a Glass of Water"')
# plt.xlabel('Time (s)')
# plt.ylabel('MFCC Coefficients')
