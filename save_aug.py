import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import random
import librosa.display
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, Gain, TimeStretch

# Specify the folder paths for input and output
input_folder = '/home/neko/new/paras/flac'
output_folder = '/home/neko/new/paras/augmented'

# Get a list of audio file names in the input folder
audio_files = os.listdir(input_folder)

# Define the augmentation transformations
augmentations = [
    ("add_gaussian_noise", AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1)),
    ("pitch_shift", PitchShift(min_semitones=-4, max_semitones=4, p=0.5)),
    ("high_pass_filter", HighPassFilter(min_cutoff_freq=500, max_cutoff_freq=3000, p=1)),
    ("random_gain", Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1)),
    ("time_stretch", TimeStretch(min_rate=0.8, max_rate=1.2, p=1))
]

# Loop over each audio file in the input folder
for audio_file in audio_files:
    # Construct the full path for the input audio file
    audio_path = os.path.join(input_folder, audio_file)

    # Load the audio signal and sample rate
    print(audio_path)
    signal, sample_rate = librosa.load(audio_path)

    # Apply augmentations and save the augmented files
    for augmentation_name, augmentation in augmentations:
        if augmentation_name == "time_shift":
            # Apply time shifting using librosa
            time_shift_factor = random.uniform(-0.5, 0.5)
            time_shifted_signal = librosa.effects.time_shift(signal, int(sample_rate * time_shift_factor))
            augmented_signal = time_shifted_signal
        else:
            augmented_signal = augmentation(samples=signal, sample_rate=sample_rate)

        # Create a folder for the specific augmentation if it doesn't exist
        augmentation_folder = os.path.join(output_folder, augmentation_name)
        os.makedirs(augmentation_folder, exist_ok=True)

        # Save the augmented files to the output folder
        filename = os.path.splitext(audio_file)[0]  # Extract the file name without extension
        out=output_folder+"/"+augmentation_name
        output_path = os.path.join(out, f"{augmentation_name}_{filename}_{augmentation_name}.flac")
        sf.write(output_path, augmented_signal, sample_rate)

        print(f"Saved augmented file: {output_path}")


files = sorted(filter( lambda x: os.path.isfile(os.path.join(directory, x)),
                       os.listdir(directory) ) )

# Get a list of file names in the folder

files = os.listdir(directory)
# Rename the files sequentially with numerical names
for i, file in enumerate(files):
    # Construct the new file name
    extension = os.path.splitext(file)[1]
    new_file_name = f"23-567244-{i+1:04d}{extension}"  # Use a 4-digit numerical name, e.g., 0001, 0002, ...

    # Construct the full paths for the old and new file names
    old_file_path = os.path.join(directory, file)
    new_file_path = os.path.join(directory , new_file_name)

    # Rename the file
    os.rename(old_file_path, new_file_path)

    print(f"Renamed file: {file} --> {new_file_name}")
