import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

def find_dat_and_bip_files():
    """
    Searches for .dat files in the "labeled_data" directory and corresponding .bip@ files 
    in the "raw_data" directory. 

    Returns:
    - dat_files_paths (list): List of relative paths to .dat files.
    - bip_files_paths (list): List of relative paths to corresponding .bip@ files.
    """

    dat_files_paths = []
    bip_files_paths = []

    bip_dir1 = []
    bip_dir2 = []

    os.chdir("labeled_data")

    for element in os.listdir():
        if os.path.isdir(element):
            os.chdir(element)
            for file in os.listdir():
                if file.endswith(".dat"):
                    dat_files_paths.append(os.path.join(os.getcwd(), file)[30:])
                    index_name_start_dir2 = file.find("_")
                    index_name_end_dir2 = file.find("-l1a")

                    bip_dir1.append(file[:index_name_start_dir2])
                    bip_dir2.append(file[:index_name_end_dir2])

            os.chdir("..")

    os.chdir("..")
    os.chdir("raw_data")

    for element in os.listdir():
        if os.path.isdir(element):
            os.chdir(element)

            for element1 in os.listdir():
                if element1 in bip_dir2:
                    os.chdir(element1)
                    for file in os.listdir():
                        if file.endswith(".bip@"):
                            bip_files_paths.append(os.path.join(os.getcwd(), file)[30:])
                        
        os.chdir("..")
    
    os.chdir("..")

    return dat_files_paths, bip_files_paths


"""
OLD VERSION

def create_csv_file():

    dat_files_paths, bip_files_paths = find_dat_and_bip_files()

    with open('files.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["dat_files", "bip_files"])
        for i in range(len(dat_files_paths)):
            writer.writerow([dat_files_paths[i], bip_files_paths[i]])
"""

def create_csv_file():

    """
    Creates a CSV file ('files.csv') that maps .dat files to their corresponding .bip@ files.

    - Calls find_dat_and_bip_files() to retrieve paths.
    - Writes the file paths into 'train_files.csv' and 'evaluate_files.csv' with headers: ["dat_files", "bip_files"].
    """

    dat_files_paths, bip_files_paths = find_dat_and_bip_files()

    if len(dat_files_paths) == 1:
        print("a")
    elif len(dat_files_paths) > 1:
        split_index = max(1, int(len(dat_files_paths) * 0.8))
        train_dat_files = dat_files_paths[:split_index]
        eval_dat_files = dat_files_paths[split_index:]
        train_bip_files = bip_files_paths[:split_index]
        eval_bip_files = bip_files_paths[split_index:]

        print(len(train_bip_files))
        print(len(train_dat_files))

        with open('train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files"])
            for i in range(len(train_dat_files)):
                writer.writerow([train_dat_files[i], train_bip_files[i]])

        with open('evaluate_files.csv', mode='w') as eval_file:
            writer = csv.writer(eval_file)
            writer.writerow(["dat_files", "bip_files"])
            for i in range(len(eval_dat_files)):
                writer.writerow([eval_dat_files[i], eval_bip_files[i]])
    else:
        with open('train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files"])
            for i in range(len(dat_files_paths)):
                writer.writerow([dat_files_paths[i], bip_files_paths[i]])

create_csv_file()

def read_csv_file(csv_file_with_path):
    """
    Reads a CSV file containing .dat and .bip@ file paths.

    Parameters:
    - csv_file_with_path (str): Path to the CSV file.

    Returns:
    - bip_files (list): List of .bip@ file paths.
    - dat_files (list): List of .dat file paths.
    """
    
    dat_files = []
    bip_files = []

    with open(f'{csv_file_with_path}', mode='r') as file:
        csvFile = csv.reader(file)

        i = 0
        for lines in csvFile:
            if i != 0:
                dat_files.append(lines[0])
                bip_files.append(lines[1])
            i += 1
    
    return bip_files, dat_files