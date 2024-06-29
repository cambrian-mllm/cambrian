import os

def remove_non_post_files(directory):
    for filename in os.listdir(directory):        
        if not filename.startswith('post'):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")

directory_path = 'topics'
remove_non_post_files(directory_path)

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith('post_'):
            new_filename = filename[5:]  
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")

rename_files(directory_path)
