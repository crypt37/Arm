import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
import librosa.display
import librosa
from matplotlib import rcParams
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter

rcParams['font.size'] = 15

audio_paths = [
    '/media/hard-drive/nlp/flac/child_male.wav',
    '/media/hard-drive/nlp/flac/Adult_male.wav'

]

# audio_paths = [
#     '/media/hard-drive/nlp/flac/child_fem.wav',
#     '/media/hard-drive/nlp/flac/adult_female.wav'
#
# ]

titles = ['Normal Signal', ' Spectrogram', 'Mel Spectrogram']

age = ["age[16-18]", "age[18-24]"]

fig, axs = plt.subplots(2, 2, figsize=(19, 9))


for i, audio_path in enumerate(audio_paths):
    signal, sampling_rate = librosa.load(audio_path, sr=None)



    # Normal Signal
    axs[i, 0].plot(signal)
    axs[i, 0].set_title(age[i])
    axs[i, 0].set_xlabel('Sample')
    axs[i, 0].set_ylabel('Amplitude')

    if i == 2:

        signal2, sampling_rate = librosa.load(audio_paths[0], sr=None)


    else:
        signal2, sampling_rate = librosa.load(audio_paths[i - 1], sr=None)



    xcorr = np.correlate(signal, signal2, mode='full')
    lag = np.arange(-len(signal) + 1, len(signal2))

    # Plot cross-correlation
    if i == 2:
        axs[i, 1].plot(lag, xcorr, label=f'Cross-correlation between {age[i]} and {age[0]}')
    else:
        axs[i, 1].plot(lag, xcorr, label=f'Cross-correlation between {age[i]} and {age[i - 1]}')

    axs[i, 1].set_xlabel('Lag')
    axs[i, 1].set_ylabel('Correlation')

    axs[i, 1].legend()

plt.subplots_adjust(wspace=0.36, hspace=0.4)
plt.tight_layout()
plt.savefig("/home/neko/aug_comp_male_laatest",dpi=200)
plt.show()
