import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

def create_csv_file():
    """
    Creates a CSV file ('files.csv') that maps .dat files 
    to their corresponding .bip (/.bip@) and .png files.

    - Calls get_dat_png_bip() to retrieve paths.
    - Writes the file paths into 'train_files.csv' and 
    'evaluate_files.csv' with headers: ["dat_files", "bip_files", "png_files"].
    """

    dat_files_paths, png_files_paths, bip_files_paths = get_dat_png_bip()

    dat_files_paths = sorted(dat_files_paths)
    png_files_paths = sorted(png_files_paths)
    bip_files_paths = sorted(bip_files_paths)

    #print(len(dat_files_paths))
    #print(dat_files_paths)

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

#######################################################################################

def read_csv_file(csv_file_with_path):
    """
    Reads a CSV file containing .dat and .bip@ file paths.

    Parameters:
    - csv_file_with_path (str): Path to the CSV file.

    Returns:
    - bip_files (list): List of .bip (/.bip@) file paths.
    - dat_files (list): List of .dat file paths.
    - png_files (list): List of .png file paths.
    """
    
    dat_files = []
    bip_files = []
    png_files = []

    with open(f'{csv_file_with_path}', mode='r') as file:
        csvFile = csv.reader(file)

        i = 0
        for lines in csvFile:
            if i != 0:
                dat_files.append(lines[0])
                bip_files.append(lines[1])
                png_files.append(lines[2])
            i += 1
    
    return bip_files, dat_files, png_files


#######################################################################################

"""
The functions move_forward(), move_back() and list_contents() are used for retrieving
.dat, .bip (/.bip@) and .png files for use in CSV creation.
"""

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

#######################################################################################

def get_dat_png_bip():
    """
    Collects and returns paths of .dat, .png, and .bip files from specified directories.
    This function navigates through a directory structure to find and collect paths of files
    with specific extensions (.dat, .png, .bip) from labeled and raw data directories.
    Returns:
        tuple: A tuple containing three lists:
            - dat_paths (list): List of paths to .dat files.
            - png_paths (list): List of paths to .png files.
            - bip_paths (list): List of paths to .bip files.
    """

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
            if "CORRECTED.dat" in j:
                dat_paths.append((str(current_path / j))[30:])
                dat_paths_checkup.append(j[:-26])
        move_back()
    
    move_back()

    #dat_paths_checkup = [str(path) for path in dat_paths_checkup]
    dat_paths_checkup = [str(path[:-10]) for path in dat_paths_checkup]
    print(dat_paths_checkup)

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

    print(len(dat_paths), len(png_paths), len(bip_paths))

    return dat_paths, png_paths, bip_paths

#######################################################################################

def read_labels_and_categorize_errors():
    """
    Reads and compares classification labels from various files and encodes discrepancies 
    between the expected and actual labels.

    The process consists of the following steps:

    1. **Initializing standard classification order**:
       - `correct_order`: Mapping for expected labels when there are 3 classes (Cloud, Land, Sea).
       - `correct_order_transform`: Mapping for expected labels when there are 4 classes (Snow, Cloud, Land, Sea).

    2. **Reading files**:
       - Reads `train_files.csv` and `evaluate_files.csv` to extract `.dat` files containing classification data.
       - Generates corresponding `.hdr` files.

    3. **Parsing `.hdr` files**:
       - Searches for lines containing `Unclassified` to retrieve the class labels used in the `.dat` files.

    4. **Reading `.dat` files**:
       - Reads binary data from `.dat` files and extracts unique labels.

    5. **Mapping labels to classes**:
       - Creates a `dict_of_order` where unique values from the `.dat` file are mapped to their corresponding classes.

    6. **Comparison and error encoding**:
       - Iterates through `dict_of_order` and compares it with `correct_order` or `correct_order_transform`.
       - If a class is misclassified, an error code is assigned based on the discrepancy:
         - `10` for `CLOUD -> LAND`
         - `11` for `CLOUD -> SEA`
         - `12` for `CLOUD -> SNOW`
         - `20` for `LAND -> CLOUD`
         - `21` for `LAND -> SEA`
         - `22` for `LAND -> SNOW`
         - `30` for `SEA -> CLOUD`
         - `31` for `SEA -> LAND`
         - `32` for `SEA -> SNOW`
         - `45` for `SNOW` (additional category in the 4-class system)
       - If no errors are detected, `0` is added.

    7. **Storing error codes**:
       - A 2D list (`list_of_indicies_of_wrong_labels_2D`) is built for all `.dat` files.

    Finally, `list_of_indicies_of_wrong_labels_2D` is returned.
    """

    correct_order = {1: "Cloud",
                     2: "Land",
                     3: "Sea"
                     }
    
    correct_order_transform = {1: "Snow",
                               2: "Cloud",
                               3: "Land",
                               4: "Sea"}

    list_of_orders_0 = []
    list_of_orders = []
    
    bip_files, dat_files, png_files = read_csv_file("train_files.csv")
    bip_files_new, dat_files_new, png_files_new = read_csv_file("evaluate_files.csv")
    bip_files.extend(bip_files_new)
    dat_files.extend(dat_files_new)
    png_files.extend(png_files_new)

    hdr_files = []
    for i in dat_files:
        hdr_files.append(f"{i[:-3]}hdr")
    
    for i in range(len(dat_files)):
        dat_file = dat_files[i][13:]
        index_slash  = dat_file.find("/")
        index_l1a = dat_file.find("-l1a")
        corr_dat_file = dat_file[index_slash+1:index_l1a]
        with open(hdr_files[i], 'r') as f_2:
            hdr = f_2.read()
            for line in hdr.split('\n'):
                if "Unclassified" in line:
                    corr_line = line[len("Unclassified, ")+1:-1]
                    corr_line_list = corr_line.split(",")
                    list_of_orders_0.append(corr_line_list)
                    #dict_of_orders[corr_dat_file] = corr_line_list

        
    #print(len(list_of_orders_0) == len(dat_files))

    for i in range(len(dat_files)):
        with open(dat_files[i], 'r') as f:
            labels = np.fromfile(f, dtype=np.uint8)
        list_of_orders.append([list_of_orders_0[i],np.unique(labels).tolist()])

    #print(list_of_orders)

    list_of_indicies_of_wrong_labels_2D = []

    for i in range(len(list_of_orders)):
        classes = None
        numbers_in_dat = None
        for j in range(len(list_of_orders[i])):
            if j % 2 == 0:
                classes = (list_of_orders[i][j])
            else:
                numbers_in_dat = (list_of_orders[i][j])
        
        #print(classes)
        #print("---")
        #print(numbers_in_dat)
        #print("#"*45)

        dict_of_order = {}

        for k in range(len(classes)):
            dict_of_order[numbers_in_dat[k]] = classes[k]
        
        #print(dict_of_order)
        combine_keys = None
        index_of_wrong_labels = 0
        list_of_indicies_of_wrong_labels = []

        #print(len(dict_of_order))

        if len(dict_of_order) == 3:
            combine_keys = correct_order.keys() & dict_of_order.keys()
            #print(combine_keys)
            for key in combine_keys:
                val1 = str(correct_order[key]).strip().upper()
                val2 = str(dict_of_order[key]).strip().upper()
                #print("#"*10)
                #print(val1, val2)
                #print(val1 == val2)
                #print("#"*10)
                if val1 != val2:
                    if val1 == "CLOUD" and val2 == "LAND":
                        list_of_indicies_of_wrong_labels.append(10)
                    if val1 == "CLOUD" and val2 == "SEA":
                        list_of_indicies_of_wrong_labels.append(11)
                    if val1 == "CLOUD" and val2 == "SNOW":
                        list_of_indicies_of_wrong_labels.append(12)
                    if val1 == "LAND" and val2 == "CLOUD":
                        list_of_indicies_of_wrong_labels.append(20)
                    if val1 == "LAND" and val2 == "SEA":
                        list_of_indicies_of_wrong_labels.append(21)
                    if val1 == "LAND" and val2 == "SNOW":
                        list_of_indicies_of_wrong_labels.append(22)
                    if val1 == "SEA" and val2 == "CLOUD":
                        list_of_indicies_of_wrong_labels.append(30)
                    if val1 == "SEA" and val2 == "LAND":
                        list_of_indicies_of_wrong_labels.append(31)
                    if val1 == "SEA" and val2 == "SNOW":
                        list_of_indicies_of_wrong_labels.append(32)                
                else:
                    list_of_indicies_of_wrong_labels.append(0)

        elif len(dict_of_order) == 4:
            combine_keys = correct_order_transform.keys() & dict_of_order.keys()
            #print(combine_keys)
            for key in combine_keys:
                val1 = str(correct_order_transform[key]).strip().upper()
                val2 = str(dict_of_order[key]).strip().upper()
                #print(val1, val2) 
                #print(val1 == val2)     
                #print("#"*10)
                if val1 != val2:
                    if val1 == "CLOUD" and val2 == "LAND":
                        list_of_indicies_of_wrong_labels.append(10)
                    if val1 == "CLOUD" and val2 == "SEA":
                        list_of_indicies_of_wrong_labels.append(11)
                    if val1 == "CLOUD" and val2 == "SNOW":
                        list_of_indicies_of_wrong_labels.append(12)
                    if val1 == "LAND" and val2 == "CLOUD":
                        list_of_indicies_of_wrong_labels.append(20)
                    if val1 == "LAND" and val2 == "SEA":
                        list_of_indicies_of_wrong_labels.append(21)
                    if val1 == "LAND" and val2 == "SNOW":
                        list_of_indicies_of_wrong_labels.append(22)
                    if val1 == "SEA" and val2 == "CLOUD":
                        list_of_indicies_of_wrong_labels.append(30)
                    if val1 == "SEA" and val2 == "LAND":
                        list_of_indicies_of_wrong_labels.append(31)
                    if val1 == "SEA" and val2 == "SNOW":
                        list_of_indicies_of_wrong_labels.append(32)
                elif val2 == "SNOW":
                    list_of_indicies_of_wrong_labels.append(45)
                else:
                    list_of_indicies_of_wrong_labels.append(0)

        list_of_indicies_of_wrong_labels_2D.append(list_of_indicies_of_wrong_labels)  
    
    return list_of_indicies_of_wrong_labels_2D


def flip_nums_in_dat(dat_path, num1, num2):
    """
    Flips occurrences of two numbers in a .dat file.

    Parameters:
    - dat_path (str): Path to the .dat file.
    - num1 (int): The first number to switch.
    - num2 (int): The second number to switch.
    """
    with open(dat_path, 'r+b') as f:
        data = np.fromfile(f, dtype=np.uint8)
        mask1 = data == num1
        mask2 = data == num2
        data[mask1] = num2
        data[mask2] = num1
        f.seek(0)
        data.tofile(f"{dat_path[:-4]}_CORR.dat")

def change_num_in_dat(dat_path, num, new_num, extra_class = False):
    """
    Changes occurrences of a number in a .dat file to a new number and subtracts 1 from all pixels.

    Parameters:
    - dat_path (str): Path to the .dat file.
    - num (int): The number to change.
    - new_num (int): The new number to replace the old number.
    """
    with open(dat_path, 'r+b') as f:
        data = np.fromfile(f, dtype=np.uint8)
        data[data == num] = new_num
        if extra_class:
            data = data - 1
        f.seek(0)
        data.tofile(f"{dat_path[:-4]}_CORR.dat")

#######################################################################################

#create_csv_file()
#change_num_in_dat("labeled_data/lacrau/lacrau_2024-12-26T11-15-54Z-l1a_products_dn_class.dat",1,2)
#change_num_in_dat("labeled_data/grizzlybay/grizzlybay_2025-01-22T19-11-18Z-l1a_products_dn_class.dat",1,2)
#change_num_in_dat("labeled_data/goddard/goddard_2025_01-09T16-07-16Z-l1a_products_dn_class.dat",1,2)
#change_num_in_dat("labeled_data/frohavet/frohavet_2025-02-25T11-26-39Z-l1a_products_dn_class.dat",1,3, extra_class=True)