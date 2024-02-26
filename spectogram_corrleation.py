import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import seaborn as sns

# Set font size for plots
plt.rcParams['font.size'] = 15

# Define audio paths
audio_paths = [
    '/media/hard-drive/nlp/flac/child_fem.wav',
    '/media/hard-drive/nlp/flac/adult_female.wav'
]
# audio_paths = [
#     '/media/hard-drive/nlp/flac/child_male.wav',
#     '/media/hard-drive/nlp/flac/Adult_male.wav'
# ]

# Define age groups
age = ["age[16-18]", "age[18-27]"]

# Initialize correlation matrix
correlation_matrix = np.zeros((len(audio_paths), len(audio_paths)))

# Compute cross-correlation between each pair of signals
for i, audio_path_1 in enumerate(audio_paths):
    for j, audio_path_2 in enumerate(audio_paths):
        # Load audio signals
        signal_1, _ = librosa.load(audio_path_1, sr=None)
        signal_2, _ = librosa.load(audio_path_2, sr=None)

        # Compute cross-correlation
        xcorr = np.correlate(signal_1, signal_2, mode='full')
        correlation_matrix[i, j] = np.max(xcorr)  # Store maximum correlation coefficient

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='hot', xticklabels=age, yticklabels=age)
plt.title('Cross-correlation Heatmap between Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Age Group')
plt.tight_layout()
plt.savefig("/home/neko/correlation_heatmap_female.png", dpi=200)
plt.show()
