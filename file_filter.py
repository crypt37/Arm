from scipy import signal
import argparse
import os
import soundfile as sf
import random
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, Gain, TimeStretch
import numpy as np

parser = argparse.ArgumentParser(description='get person name ')
parser.add_argument('--name',
                    help='name of folder ')
args = parser.parse_args()


def apply_band_pass_filter(signal_data, sampling_freq):
    low_cut_freq = 200  # Lower cutoff frequency in Hz
    high_cut_freq = 3500  # Higher cutoff frequency in Hz
    order = 4  # Filter order
    nyquist_freq = 0.5 * sampling_freq
    normalized_low_cutoff = low_cut_freq / nyquist_freq
    normalized_high_cutoff = high_cut_freq / nyquist_freq
    b, a = signal.butter(order, [normalized_low_cutoff, normalized_high_cutoff], btype='band')
    filtered_sig = signal.lfilter(b, a, signal_data)
    return filtered_sig


# Specify the input folder and output folder
input_folder = '/home/neko/new/' + args.name + "/flac"
output_folder = '/home/neko/new/' + args.name

augmentations = [
    ("add_gaussian_noise", AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5)),
    ("pitch_shift", PitchShift(min_semitones=-4, max_semitones=4, p=0.5)),
    ("high_pass_filter", HighPassFilter(min_cutoff_freq=500, max_cutoff_freq=3000, p=0.5)),
    ("random_gain", Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5)),
    ("time_stretch", TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5))
]

# Get the list of sound files in the input folder
sound_files = os.listdir(input_folder)

# Process each sound file
for sound_file in sound_files:
    # Construct the full paths for input and output files
    input_file = os.path.join(input_folder, sound_file)
    output_file = os.path.join(output_folder, sound_file)
    audio_path = os.path.join(input_folder, sound_file)
    signal_wav, sample_rate = librosa.load(audio_path)
    filtered_signal = apply_band_pass_filter(signal_data=signal_wav, sampling_freq=sample_rate)
    sf.write(output_file, filtered_signal, sample_rate)
    # Load the input audio file

    for augmentation_name, augmentation in augmentations:
        if augmentation_name == "time_shift":
            # Apply time shifting using librosa
            time_shift_factor = random.uniform(-0.5, 0.5)
            time_shifted_signal = librosa.effects.time_shift(signal_wav, int(sample_rate * time_shift_factor))
            augmented_signal = time_shifted_signal
        else:
            augmented_signal = augmentation(samples=signal_wav, sample_rate=sample_rate)

        # Create a folder for the specific augmentation if it doesn't exist
        augmentation_folder = os.path.join(output_folder, augmentation_name)
        os.makedirs(augmentation_folder, exist_ok=True)

        # Save the augmented files to the output folder
        filename = os.path.splitext(sound_file)[0]  # Extract the file name without extension
        out = output_folder + "/" + augmentation_name
        output_path = os.path.join(out, f"{augmentation_name}_{filename}_{augmentation_name}.flac")
        sf.write(output_path, augmented_signal, sample_rate)

print("Filtered sounds saved in", output_folder)

name = ["time_stretch", "random_gain", "pitch_shift", "add_gaussian_noise", "high_pass_filter", "flac"]
for dir_name in name:
    # Set the directory path
    directory = '/home/neko/new/' + args.name + '/' + dir_name
    # Get all files in the directory
    files = os.listdir(directory)
    # Specify the folder path
    files = sorted(filter(lambda x: os.path.isfile(os.path.join(directory, x)),
                          os.listdir(directory)))
    # Get a list of file names in the folder
    # Rename the files sequentially with numerical names
    random_number = random.randint(1000, 9999)
    for i, file in enumerate(files):
        # Construct the new file name
        extension = os.path.splitext(file)[1]
        new_file_name = f"15-{random_number}-{i + 1:04d}{extension}"  # Use a 4-digit numerical name, e.g., 0001, 0002, ...

        # Construct the full paths for the old and new file names
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)

        print(f"Renamed file: {file} --> {new_file_name}")

# Get a list of files in the folder
files = os.listdir(output_folder)

# Filter and delete files with the ".wav" extension
for file in files:
    if file.endswith(".flac"):
        file_path = os.path.join(output_folder, file)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")


def create_file_list(folder_path, output_file):
    try:
        with open(output_file, 'w') as file_list:
            files = sorted(os.listdir(folder_path))
            for index, file_name in enumerate(files, start=1):
                if os.path.isfile(os.path.join(folder_path, file_name)):
                    file_list.write(f"{file_name}\n")
        print("File list created successfully.")
    except IOError:
        print("An error occurred while creating the file list.")


create_file_list(input_folder, output_file="transcribe_this.txt")
