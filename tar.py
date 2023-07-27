import tarfile

# Specify the path to your tar file
tar_file_path = "/Users/cw/Desktop/properties.tar"

# Open the tar file
with tarfile.open(tar_file_path, "r") as tar:
    # List the contents of the tar file (optional)
    print("Contents of the tar file:")
    for member in tar.getmembers():
        print(member.name)

    # Extract all contents to a specific directory
    target_directory = "/Users/cw/Desktop/test"
    tar.extractall(path=target_directory)

# The tar file is automatically closed after the 'with' block


