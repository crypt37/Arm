import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
import librosa.display
import librosa
from matplotlib import rcParams
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter
rcParams['font.size'] = 15



# audio_paths = [
#     '/media/hard-drive/nlp/flac/flac/34-586334-01.flac',
#     '/media/hard-drive/nlp/flac/adult.flac',
#     '/media/hard-drive/nlp/flac/child.flac'
# ]

audio_paths = [
    '/media/hard-drive/nlp/flac/child_fem.flac',
    '/media/hard-drive/nlp/flac/adul_fem.flac',
    '/media/hard-drive/nlp/flac/old_fem.flac'
]


titles = ['Normal Signal', 'Normal Spectrogram', 'Mel Spectrogram']


age=["age[16-18]","age[18-24]","age[30-50]"]

fig, axs = plt.subplots(3, 3, figsize=(19, 9))


for i, audio_path in enumerate(audio_paths):
    signal, sampling_rate = librosa.load(audio_path, sr=None)

    # Normal Signal
    axs[i, 0].plot(signal)
    axs[i, 0].set_title(age[i])
    axs[i, 0].set_xlabel('Sample')
    axs[i, 0].set_ylabel('Amplitude')

    # Normal Spectrogram
    axs[i, 1].specgram(signal, Fs=sampling_rate, cmap='binary')
    axs[i, 1].set_title(titles[1])
    axs[i, 1].set_xlabel('Time (s)')
    axs[i, 1].set_ylabel('Frequency (Hz)')

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sampling_rate, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec)
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sampling_rate, cmap="binary", ax=axs[i, 2])
    axs[i, 2].set_xlabel("Time(s)")
    axs[i, 2].set_ylabel("Mel Frequency")
    axs[i, 2].set_title(titles[2])

plt.tight_layout()
plt.subplots_adjust(wspace=0.36, hspace=0.4)
plt.show()
