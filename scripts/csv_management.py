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
                    if "fixed.dat" in file:
                        dat_files_paths.append(os.path.join(os.getcwd(), file)[30:])
                        index_name_start_dir2 = file.find("_")
                        index_name_end_dir2 = file.find("-l1a")

                        bip_dir1.append(file[:index_name_start_dir2])
                        bip_dir2.append(file[:index_name_end_dir2])
                    elif "fixed.dat" not in file:
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
                        if file.endswith(".bip@") or file.endswith(".bip"):
                            bip_files_paths.append(os.path.join(os.getcwd(), file)[30:])
        os.chdir("..")
    
    os.chdir("..")

    for elem in dat_files_paths:
        print(f"Length of {elem}: {os.path.getsize(elem)} bytes")

    for elem in dat_files_paths:
        for elem1 in dat_files_paths:
            if "fixed" in elem:
                index_fixed = elem.find("_fixed")
                if elem[:index_fixed] == elem1[:index_fixed]:
                    index_elem1 = dat_files_paths.index(elem1)
                    dat_files_paths.pop(index_elem1)

    return dat_files_paths, bip_files_paths

#find_dat_and_bip_files()

def create_csv_file():

    """
    Creates a CSV file ('files.csv') that maps .dat files to their corresponding .bip@ files.

    - Calls find_dat_and_bip_files() to retrieve paths.
    - Writes the file paths into 'train_files.csv' and 'evaluate_files.csv' with headers: ["dat_files", "bip_files"].
    """

    dat_files_paths, png_files_paths, bip_files_paths = get_dat_png_bip()

    #print(len(dat_files_paths),len(png_files_paths),len(bip_files_paths))

    if len(dat_files_paths) == 1:
        with open('train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            writer.writerow([dat_files_paths[0], bip_files_paths[0], png_files_paths[0]])
    elif len(dat_files_paths) > 1:
        split_index = int(len(dat_files_paths)*0.6)
        train_dat_files = dat_files_paths[:split_index]
        eval_dat_files = dat_files_paths[split_index:]

        train_bip_files = bip_files_paths[:split_index]
        eval_bip_files = bip_files_paths[split_index:]

        train_png_files = png_files_paths[:split_index]
        eval_png_files = png_files_paths[split_index:]

        with open('train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            for i in range(len(train_dat_files)):
                writer.writerow([train_dat_files[i], train_bip_files[i], train_png_files[i]])

        with open('evaluate_files.csv', mode='w') as eval_file:
            writer = csv.writer(eval_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            for i in range(len(eval_dat_files)):
                writer.writerow([eval_dat_files[i], eval_bip_files[i], eval_png_files[i]])
    else:
        with open('train_files.csv', mode='w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["dat_files", "bip_files", "png_files"])
            for i in range(len(dat_files_paths)):
                writer.writerow([dat_files_paths[i], bip_files_paths[i], eval_png_files[i]])


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


#######################################################################################

current_path = Path(".").resolve()

def move_forward(folder_name):
    global current_path
    new_path = current_path / folder_name
    if new_path.exists() and new_path.is_dir():
        current_path = new_path
        return f"Moved to: {current_path}"
    return "Folder does not exist."

def move_back():
    global current_path
    if current_path.parent.exists():
        current_path = current_path.parent
        return f"Moved back to: {current_path}"
    return "Cannot move back further."

def list_contents():
    return [item.name for item in current_path.iterdir()]

def print_files():
    files = [file.name for file in current_path.iterdir() if file.is_file()]

def get_dat_png_bip():
    dat_paths = []
    dat_paths_checkup = []

    png_paths = []
    bip_paths = []

    move_forward("labeled_data")
    contents_labeled_data = list_contents()
    for i in contents_labeled_data:
        move_forward(i)
        contents_i_data = list_contents()
        for j in contents_i_data:
            if "class.dat" in j:
                dat_paths.append((str(current_path / j))[30:])
                dat_paths_checkup.append(j[:-26])
        move_back()
    
    move_back()

    dat_paths_checkup = [str(path) for path in dat_paths_checkup]

    move_forward("raw_data")
    L1_contents = list_contents()

    for folder in L1_contents:
        move_forward(folder)
        L2_contents = list_contents()
        for folder_gr_2 in L2_contents:
            if folder_gr_2 in dat_paths_checkup:
                move_forward(folder_gr_2)
                L3_contents = list_contents()
                for i in L3_contents:
                    if i.endswith("Z.png"):
                        png_paths.append((str(current_path / i))[30:])
                    elif i.endswith(".bip"):
                        bip_paths.append((str(current_path / i))[30:])
                    elif i.endswith(".bip@"):
                        bip_paths.append((str(current_path / i))[30:])

                move_back()
        move_back()
    move_back()

    return dat_paths, png_paths, bip_paths

create_csv_file()