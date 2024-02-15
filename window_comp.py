import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import kaiser, gaussian


def custom_gaussian_window(n_fft, std_dev, num_points):
    x = np.linspace(-num_points / 2, num_points / 2, num_points)
    window = np.exp(-(x / (std_dev / 2))**2)
    window = np.pad(window, (n_fft - num_points) // 2, mode='constant')
    return window

def plot_spectrogram(signal, sr, window_function, window_param=None, subplot_index=None):
    # Set the STFT window length (n_fft)
    n_fft = 2048  # You can adjust this value based on your requirements

    # Generate the window function
    if window_function == 'kaiser' and window_param is not None:
        window = kaiser(n_fft, beta=window_param)
    elif window_function == 'gaussian' and window_param is not None:
        std_dev, num_points = window_param
        window = custom_gaussian_window(n_fft, std_dev, num_points)
    else:
        window = window_function

    # Compute the Short-Time Fourier Transform (STFT) of the signal with the specified window
    D = librosa.stft(signal, n_fft=n_fft, window=window)

    # Convert the magnitude spectrogram to dB scale
    mag, _ = librosa.magphase(D)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)

    # Plot the spectrogram in a grid
    plt.subplot(2, 4, subplot_index)
    librosa.display.specshow(mag_db, sr=sr, hop_length=n_fft//4, x_axis='time', y_axis='linear', cmap='viridis')
    plt.title(window_function.capitalize())
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

if __name__ == "__main__":
    # Load the audio file
    audio_path = '/media/hard-drive/nlp/nlp_water/glassofwater.flac'
    signal, sr = librosa.load(audio_path)

    # Define a list of window functions and their parameters (if applicable)
    window_functions = [
        ('rectangular', None),
        ('bartlett', None),
        ('hamming', None),
        ('blackman', None),
        ('kaiser', 5),  # Beta parameter for the Kaiser window (can be adjusted)
        ('gaussian', (100, 100)),  # Tuple of (std_dev, num_points) for the Gaussian window
        ('hann', None),
        ('triang', None)
    ]

    # Create a single figure for the grid of subplots
    plt.figure(figsize=(14, 8))

    # Create spectrograms using different window functions and plot in the grid
    for i, (window_function, window_param) in enumerate(window_functions, start=1):
        plot_spectrogram(signal, sr, window_function, window_param, i)
    # Add a suptitle to the figure
    plt.suptitle("Spectrograms with Different Window Functions", fontsize=16)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
