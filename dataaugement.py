import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

import random
import librosa.display
from scipy.io import wavfile
from pydub import AudioSegment
import pydub.effects as effects
from scipy.signal import spectrogram

audio_path = 'sound.wav'

signal, sample_rate, = librosa.load(audio_path)

frame_size=2048
n_fft = 1024
hop_length = 512
n_mels = 80

plt.plot(signal)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'Amplitude Normalized Waveform for  {audio_path[4:-4]}')
plt.grid(True)
audio =signal
plt.plot(signal)
speed_factor = random.uniform(0.9, 1.1)
pitch_shift = random.randint(-2, 2)
augmented_audio = audio._spawn(audio.raw_data, overrides={
    "frame_rate": int(audio.frame_rate * speed_factor),
    "pitch": pitch_shift
})

plt.plot(signal)