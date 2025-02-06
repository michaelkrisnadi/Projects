import os
import shutil

def join_files(source_directory, target_directory, course_codes):
    source_dir = source_directory
    target_dir = target_directory

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            new_filename = old_path.replace(source_dir, "").replace(os.sep, "-").lstrip("-")
            new_path = os.path.join(target_dir, new_filename)
            shutil.copy(old_path, new_path)

    print("All files have been copied.")

    # Only keep files that has the course code in the array
    directory = target_dir
    if 'all' in course_codes:
        course_codes = ['COM102', 'COM106', 'COM103', 'COM107', 'COM204', 'COM205', 'EEE202', 'EEE203', 'COM203', 'COM207', 'EEE301', 'EEE313', 'EEE401', 'EEE302', 'EEE303', 'EEE304', 'EEE316', 'EEE319', 'EEE305', 'EEE308', 'EEE309', 'EEE322', 'EEE310', 'EEE404', 'EEE406', 'EEE407', 'EEE411', 'EEE414', 'EEE405']

    # Delete file outside the course list
    for filename in os.listdir(directory):
        if not any(code in filename or code.replace(" ", "") in filename for code in course_codes):
            os.remove(os.path.join(directory, filename))

    print("Files have been filtered.")