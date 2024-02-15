import numpy as np
import matplotlib.pyplot as plt
import librosa

# Load the audio file
audio_path = 'sound.wav'
waveform, sample_rate = librosa.load(audio_path, sr=None)

# Compute the STFT
window_size = 1024
hop_length = 512
stft = librosa.stft(waveform, n_fft=window_size, hop_length=hop_length)

# Set the number of frames to plot
num_frames = 5

# Select the first few frames from the waveform
waveform_frames = waveform[:num_frames * hop_length]

# Select the first few frames from the STFT
stft_frames = stft[:, :num_frames]

# Create a figure and axes
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot the original waveform frames
axs[0].plot(waveform_frames)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Original Waveform')

# Plot the stacking of windows for the STFT frames
for i in range(num_frames):
    window = stft_frames[:, i]
    time_indices = np.arange(i * hop_length, i *window_size/2+hop_length+1)

    axs[1].plot(time_indices, window, label=f"Frame {i+1}")

# Set the x-axis label
axs[1].set_xlabel('Time')

# Set the y-axis label
axs[1].set_ylabel('Amplitude')

# Set the plot title
axs[1].set_title('Stacking of Windows in STFT')

# Add a legend
axs[1].legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()
