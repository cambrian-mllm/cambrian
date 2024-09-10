import os


def remove_non_post_files(directory):
    for filename in os.listdir(directory):
        if not filename.startswith('post'):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")


def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith('post_'):
            new_filename = filename[5:]
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Clean and rename topics JSON files in a directory')
    parser.add_argument('--topics_dir', type=str, default='./data/topics/',
                        help='Directory of topics JSON files to process')
    args = parser.parse_args()

    remove_non_post_files(args.topics_dir)
    rename_files(args.topics_dir)
