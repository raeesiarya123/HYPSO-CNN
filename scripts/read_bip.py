import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

# make rgb from lidsat
# code to overlap HSI and RGB captures most effectively

DEBUG = True

def read_bip(top_folder_name, lines=965, samples=684, bands=120, rband=59, gband=70, bband=89, flip=False, binning=1, scaling=1.0):
    '''Code taken from make-rgb.py and h1data_processing.py (get_metainfo and get_raw_cube functions) 
    '''
    info = {}
    config_file_path = os.path.join(
     top_folder_name, "capture_config.ini")
    

    def is_integer_num(n) -> bool:
        if isinstance(n, int):
            return True
        if isinstance(n, float):
            return n.is_integer()
        return False

    # read all lines in the config file
    with open(config_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # split the line at the equal sign
            line = line.split("=")
            # if the line has two elements, add the key and value to the
            # info dict
            if len(line) == 2:
                key = line[0].strip()
                value = line[1].strip()
                try:
                    if is_integer_num(float(value)):
                        info[key] = int(value)
                    else:
                        info[key] = float(value)
                except BaseException:
                    info[key] = value

    info["image_height"] = info["row_count"]
    info["image_width"] = int(info["column_count"] / info["bin_factor"])
    info["im_size"] = info["image_height"] * info["image_width"]
    
    # Read input image
    path_to_bip = os.path.join(
            top_folder_name, 'z_compressed_cube.bip')

    cube = np.fromfile(path_to_bip, dtype='uint16')
    if DEBUG:
        print(path_to_bip)
        print(cube.shape)
    cube = cube.reshape(
        (-1, info["image_height"], info["image_width"]))

    if binning > 1:
        cube = cube[:,::info["bin_factor"],:]

    composite = cube[:,:,(rband, gband, bband)].astype('float32')
    if flip:
        composite = composite[:,::-1,:]
    composite = composite.transpose((1,0,2))


    composite_8bit = (255*composite/composite.max())

    image = Image.fromarray(composite_8bit.astype('uint8')) # needs to be uint8 datatype

    w, h = image.size
    original_ar = w/h
    if scaling > 1.0:
        new_w = int(w*scaling)
        new_h = h
    else:
        new_w = w
        new_h = int(h/scaling)

    image = image.resize((new_w, new_h))#, resample=Image.Resampling.LANCZOS)
    # Image.Resample does not exist in 7.0.0, which is the verion of Pillow on the lidsat

    return image, cube

# image, cube = make_rgb('data/plocan_2024_12_18T11_59_59', scaling=8)
# plt.figure(figsize=(15,6))
# plt.imshow(image)
# plt.axis("off")
# plt.figure(figsize=(15,6))
# composite = cube[:,:,(59, 70, 89)].astype('float32')
# composite = composite.transpose((1,0,2))
# composite_8bit = (255*composite/composite.max())
# image = Image.fromarray(composite_8bit.astype('uint8'))
# plt.imshow(image)
# plt.axis("off")

top_folder_name = 'collected_data/aeronetgalata/aeronetgalata_2025-01-02T08-52-34Z-hsi0'

print(read_bip(top_folder_name, scaling=8))