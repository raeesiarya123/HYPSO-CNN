import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

def find_dat_and_bip_files():    
    dat_files_names = []
    dat_files_paths = []

    bip_dir1 = []
    bip_dir2 = []

    bip_files_paths = []

    os.chdir("training_data")

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


def create_csv_file():
    dat_files_paths, bip_files_paths = find_dat_and_bip_files()

    with open('files.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["dat_files", "bip_files"])
        for i in range(len(dat_files_paths)):
            writer.writerow([dat_files_paths[i], bip_files_paths[i]])

#create_csv_file()

def read_csv_file():
    dat_files = []
    bip_files = []

    with open('files.csv', mode='r') as file:
        csvFile = csv.reader(file)

        i = 0
        for lines in csvFile:
            if i != 0:
                dat_files.append(lines[0][1:])
                bip_files.append(lines[1][1:])
            i += 1
    
    return bip_files, dat_files