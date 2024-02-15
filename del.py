import os

folder_path = '/home/neko/new/filtered'  # Specify the folder path

# Get a list of files in the folder
files = os.listdir(folder_path)

# Filter and delete files with the ".wav" extension
for file in files:
    if file.endswith(".wav"):
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
