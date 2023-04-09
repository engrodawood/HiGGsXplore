
#*****************************************************
# Patches from the wsi are extracted using TIA-TOOLBOX
# Tiles are extracted at 40x 256x256 for cellular composition inference using 
# ALBRT, as that model is trained on patches of size 256x256 at 40x. 

# For obtaining deep features patches of size 512x512 at 20x are extracted. 
#******************************************************

from tiatoolbox.wsicore import wsireader
from tiatoolbox.tools import patchextraction
import os
from skimage.io import imsave
from tqdm import tqdm
import numpy as np
import pandas as pd

# Extracting Patches for Cellular Composition Inference
ALBRT = False
patch_size = (512,512)
MPP = 0.50
if ALBRT:
    patch_size = (256,256)
    MPP = 0.25

# Base directory of project
bdir = '/data/PanCancer/HTEX_repo'
DATA_DIR = f'{bdir}/data'

mask_path = f'{DATA_DIR}/mask'
svs_data_dir = f'{DATA_DIR}/WSIs'
TAG = 'x'.join(map(str,patch_size))

tiles_path = f'{DATA_DIR}/Patches/{TAG}'

def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def contains_tissue(im, color_threshold=200, percentage_threshold=0.6):
    '''
    If more than 60% pixels have mean value above 200 ignore those. 
    '''
    return np.mean(np.mean(im, axis=2).flatten() > color_threshold)<percentage_threshold


def patchify(patch,patch_id):
    if contains_tissue(patch):
        patch_name = str(PE.locations_df['x'][patch_id])+'_'+str(PE.locations_df['y'][patch_id])+'.jpg'
        imsave(otiles+'/'+patch_name, patch)

selectedPatients = pd.read_csv(f'{DATA_DIR}/selected_WSI.csv',index_col='Patient ID')
# import pdb; pdb.set_trace()
# Loading WSIs
pids = os.listdir(svs_data_dir)
for pid in tqdm(pids):
    if os.path.isfile(os.path.join(svs_data_dir,pid)):continue
    for filename in os.listdir(os.path.join(svs_data_dir,pid)):
        patient_id = filename.split('.')[0]
        if filename.endswith('svs'):
            wsi_path = os.path.join(svs_data_dir,pid,filename)
            print('processing... ',patient_id)  
            otiles = os.path.join(tiles_path,patient_id)

            # Use only selected patients
            if patient_id not in selectedPatients['WSI Name'].tolist(): continue
            # If directory with WSI nake is already available skip PE
            if os.path.isdir(otiles):
                continue
            mkdirs(otiles)

            print("Tiles saving dir ",otiles)
            wsi = wsireader.get_wsireader(input_img=wsi_path)
            mask_name = f'{mask_path}/{filename.replace("svs","png")}'

            # Checking for Tissue mask, if tissue mask is not available skip PE
            # comment this line if you prefer not using tissue mask
            if not os.path.isfile(mask_name):
                continue
            PE = patchextraction.get_patch_extractor(
            input_img=wsi, # input image path, numpy array, or WSI object
            method_name="slidingwindow", units = 'mpp',resolution=MPP,input_mask=mask_name,
            patch_size=patch_size, # size of the patch to extract around the centroids from centroids_list
            stride=patch_size) # stride of extracting patches, default is equal to patch_size  

            for idx,patch in enumerate(tqdm(PE)):
                patchify(patch,idx)