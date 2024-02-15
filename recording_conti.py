import os
import sounddevice as sd
import numpy as np
import time
from scipy.io import wavfile
import soundfile as sf

# Set the audio parameters
sample_rate = 44100  # Sample rate in Hz
duration = 0.5  # Duration of each audio segment in seconds
threshold_energy = -30  # Energy threshold for voice activity detection (converted from dB to energy value)
threshold_silence = 3 # Silence threshold in seconds
output_folder = "/media/hard-drive/final_sqee/datasets/LibriSpeech/dev-test/12/212122/"  # Folder to store the recorded audio
eval_file = "/media/hard-drive/final_sqee/LibriSpeech/eval.txt"  # File to store the evaluation information
folder = "dev-test/12/212122/"

def calculate_energy(audio_segment):
    # Calculate the energy of an audio segment
    return np.sum(np.abs(audio_segment) ** 2) / len(audio_segment)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Start recording audio
print("Recording started. Press Ctrl+C to stop...")
duration_thres=3
is_recording = False
start_time = 0
file_counter = 1
recording = []
threshold_stop=-40
try:
    while True:
        # Start recording the audio segment
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        # Calculate the energy of the current audio segment
        energy = calculate_energy(audio)
        energy_db = 10 * np.log10(energy)  # Convert energy value to dB
        print(energy_db)

        transcript = "SAMPLE TRANS"
        label = "4 4 4 4 4 4 4"

        if energy_db > threshold_energy:

            if not is_recording:
                print("Voice detected")
                is_recording = True
                start_time = time.time() - duration_thres  # Adjust start time by subtracting duration
                file_path = os.path.join(output_folder, f"{file_counter}.flac")
                file_counter += 1

            recording.append(audio)

        elif is_recording and energy_db <= threshold_stop:
            print("Recording stopped")
            is_recording = False
            recording.append(audio)
            # Convert the recording list to a numpy array

            recording = np.concatenate(recording)
            recorded = True

            # Save the recorded audio to a file
            wavfile.write(file_path, sample_rate, recording)
            sf.write(file_path, recording, sample_rate)

            # Reset the recording list
            recording = []

            # Append the location of the recording to the evaluation file
            with open(eval_file, "a") as f:  # Open the file in "append" mode
                f.write(f"{file_path}\t{transcript}\t{label}\n")


except KeyboardInterrupt:
    print("Recording interrupted. Exiting...")
    sd.stop()