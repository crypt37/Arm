import os
import sounddevice as sd
import numpy as np
import time
from scipy.io import wavfile
import soundfile as sf

# Set the audio parameters
sample_rate = 44100  # Sample rate in Hz
duration = 0.25  # Duration of each audio segment in seconds

threshold_energy = -33  # Energy threshold for voice activity detection (converted from dB to energy value)
output_folder = "/media/hard-drive/final_sqee/datasets/LibriSpeech/dev-test/12/212122/"  # Folder to store the recorded audio
eval_file = "/media/hard-drive/final_sqee/LibriSpeech/eval.txt"  # File to store the evaluation information
thres_stop_energy=-45
def calculate_energy(audio_segment):
    # Calculate the energy of an audio segment
    return np.sum(np.abs(audio_segment) ** 2) / len(audio_segment)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Start recording audio
print("Recording started. Press Ctrl+C to stop...")
recording = []
is_recording = False
start_time = 0
file_counter = 1
start_thres=1.5
stop_thres=2
try:
    while True:
        # Start recording the audio segment
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        # Calculate the energy of the current audio segment
        energy = calculate_energy(audio)
        energy_db = 10 * np.log10(energy)  # Convert energy value to dB
        print(energy_db)

        if energy_db > threshold_energy:
            if not is_recording:
                print("Voice detected")
                is_recording = True
                start_time = time.time()-start_thres

        elif is_recording:
            if time.time() - start_time >= stop_thres and energy_db<=thres_stop_energy:
                print("Recording stopped")
                is_recording = False

                # Convert the recording list to a numpy array
                recording = np.concatenate(recording)

                # Save the recorded audio to a file
                file_path = os.path.join(output_folder, f"{1}.flac")
                wavfile.write(file_path, sample_rate, recording)
                sf.write(file_path, recording, sample_rate)

                # Append the location of the recording to the evaluation file
                transcript = "SAMPLE TRANS"
                label = "4 4 4 4 4 4 4"
                with open(eval_file, "w") as f:  # Open the file in "append" mode
                    f.write(f"{file_path}\t{transcript}\t{label}\n")

                # Reset variables for the next recording
                recording = []
                file_counter += 1

        if is_recording:
            recording.append(audio)

except KeyboardInterrupt:
    print("Recording interrupted. Exiting...")
    sd.stop()
