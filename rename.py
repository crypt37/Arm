import os

# Set the directory path
directory = '/home/neko/new/jug/augmented/time_stretch'

# Get all files in the directory
files = os.listdir(directory)

# Sort the files in alphabetical order


# Rename files with serial numbers
# for i, file_name in enumerate(files):
#     # Create the new file name with serial number
#     new_file_name = f"{i+1:03d}.flac"  # Adjust the format as needed
#
#     # Construct the full paths of the old and new file names
#     old_path = os.path.join(directory, file_name)
#     new_path = os.path.join(directory, new_file_name)
#
#     # Rename the file
#     os.rename(old_path, new_path)


# Specify the folder path
files = sorted(filter( lambda x: os.path.isfile(os.path.join(directory, x)),
                        os.listdir(directory) ) )

# Get a list of file names in the folder


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
